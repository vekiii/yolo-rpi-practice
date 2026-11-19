#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model_builder.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h>
#include <tensorflow/lite/version.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using json=nlohmann::json;
using Clock=std::chrono::steady_clock;
namespace fs=std::filesystem;
struct Detection {float x1,y1,x2,y2,score; int cls;};
static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
static float overlap(const Detection&a,const Detection&b){
 float w=std::max(0.f,std::min(a.x2,b.x2)-std::max(a.x1,b.x1));
 float h=std::max(0.f,std::min(a.y2,b.y2)-std::max(a.y1,b.y1));
 float u=(a.x2-a.x1)*(a.y2-a.y1)+(b.x2-b.x1)*(b.y2-b.y1)-w*h;
 return u>0?w*h/u:0;
}
static void prepare(const cv::Mat&img,float* input,bool nhwc,float&r,int&left,int&top){
 r=std::min(640.f/img.cols,640.f/img.rows);
 int w=int(std::nearbyint(img.cols*r)),h=int(std::nearbyint(img.rows*r));
 left=int(std::nearbyint((640-w)/2.0-0.1));top=int(std::nearbyint((640-h)/2.0-0.1));
 cv::Mat resized,padded,rgb;
 cv::resize(img,resized,cv::Size(w,h),0,0,cv::INTER_LINEAR);
 cv::copyMakeBorder(resized,padded,top,640-h-top,left,640-w-left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
 cv::cvtColor(padded,rgb,cv::COLOR_BGR2RGB);
 for(int y=0;y<640;y++)for(int x=0;x<640;x++)for(int c=0;c<3;c++){
  int pixel=y*640+x;
  input[nhwc?pixel*3+c:c*640*640+pixel]=float(rgb.ptr<unsigned char>(y)[x*3+c])/255.f;
 }
}
static std::vector<Detection> decode(const float* data,bool features_first,bool normalized,float ratio,int left,int top,int width,int height){
 auto at=[&](int f,int a){return features_first?data[f*8400+a]:data[a*84+f];};
 std::vector<Detection> candidates;
 for(int a=0;a<8400;a++){
  int cls=0;float score=at(4,a);
  for(int c=1;c<80;c++)if(at(c+4,a)>score){score=at(c+4,a);cls=c;}
  if(score<0.001f)continue;
  float scale=normalized?640.f:1.f;
  float cx=at(0,a)*scale,cy=at(1,a)*scale,w=at(2,a)*scale,h=at(3,a)*scale;
  candidates.push_back({cx-w/2,cy-h/2,cx+w/2,cy+h/2,score,cls});
 }
 std::stable_sort(candidates.begin(),candidates.end(),[](auto&a,auto&b){return a.score>b.score;});
 std::vector<Detection> keep;
 for(auto d:candidates){
  bool suppressed=false;
  for(const auto&k:keep)if(d.cls==k.cls&&overlap(d,k)>0.7f){suppressed=true;break;}
  if(!suppressed){keep.push_back(d);if(keep.size()==100)break;}
 }
 for(auto&d:keep){
  d.x1=std::clamp((d.x1-left)/ratio,0.f,float(width)); d.x2=std::clamp((d.x2-left)/ratio,0.f,float(width));
  d.y1=std::clamp((d.y1-top)/ratio,0.f,float(height)); d.y2=std::clamp((d.y2-top)/ratio,0.f,float(height));
 }
 return keep;
}
static std::vector<int> shape(const TfLiteTensor*t){return {t->dims->data,t->dims->data+t->dims->size};}
static json precision_metadata(const tflite::Model& graph, const TfLiteTensor& input,
                               const TfLiteTensor& output) {
 std::map<std::string, size_t> tensor_counts;
 for (const auto* subgraph : *graph.subgraphs()) {
  if (!subgraph->tensors()) continue;
  for (const auto* tensor : *subgraph->tensors()) {
   const char* name = tflite::EnumNameTensorType(tensor->type());
   ++tensor_counts[*name ? name : "UNKNOWN_" + std::to_string(static_cast<int>(tensor->type()))];
  }
 }
 return {
  {"input_dtype", TfLiteTypeGetName(input.type)},
  {"output_dtype", TfLiteTypeGetName(output.type)},
  {"graph_tensor_counts_by_type", tensor_counts},
  {"runtime_kernel_precision", {{"status", "not_profiled"}}}
 };
}
static void check(TfLiteStatus s,const char* message){if(s!=kTfLiteOk)throw std::runtime_error(message);}
struct Stats {double mean,median,std,min,max;};
static Stats stats(std::vector<double> v){
 if(v.empty())throw std::runtime_error("Cannot summarize empty timing data");
 double sum=0;for(double x:v)sum+=x;double mean=sum/v.size(),sq=0;
 for(double x:v){double d=x-mean;sq+=d*d;}
 std::sort(v.begin(),v.end());size_t n=v.size();double median=n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;
 return {mean,median,std::sqrt(sq/n),v.front(),v.back()};
}
static json statsJson(const std::vector<double>&v){auto s=stats(v);return {{"mean",s.mean},{"median",s.median},{"std",s.std},{"min",s.min},{"max",s.max}};}
int main(int argc,char**argv){try{
 if(argc>=2&&std::string(argv[1])=="--benchmark"){
  if(argc<8){std::cerr<<"Usage: yolo_tflite --benchmark MODEL.tflite IMAGES MANIFEST OUTPUT_PREFIX THREADS normalized|pixels [xnnpack|builtin] [--seed N] [--image ID] [--warmup N] [--iterations N]\n";return 1;}
  fs::path model_path=argv[2],images=argv[3],prefix=argv[5];int threads=std::stoi(argv[6]);std::string units=argv[7],backend="xnnpack";
  uint32_t seed=1;int warmup=50,iterations=5000;bool explicit_image=false;json selected_id;
  for(int i=8;i<argc;i++){std::string a=argv[i];if(a=="--seed"&&i+1<argc)seed=static_cast<uint32_t>(std::stoul(argv[++i]));else if(a=="--image"&&i+1<argc){selected_id=std::stoll(argv[++i]);explicit_image=true;}else if(a=="--warmup"&&i+1<argc)warmup=std::stoi(argv[++i]);else if(a=="--iterations"&&i+1<argc)iterations=std::stoi(argv[++i]);else if((a=="xnnpack"||a=="builtin")&&i==8)backend=a;else throw std::runtime_error("Invalid benchmark option: "+a);}
  if(threads<1||warmup<0||iterations<1||(units!="normalized"&&units!="pixels"))throw std::runtime_error("Invalid benchmark configuration");
  std::ifstream mf(argv[4]);if(!mf)throw std::runtime_error("Cannot open manifest");json manifest;mf>>manifest;if(manifest.at("images").empty()||manifest.at("category_ids").size()!=80)throw std::runtime_error("Invalid manifest");
  json item;uint32_t actual_seed=seed;
  if(explicit_image){bool found=false;for(const auto&x:manifest["images"])if(x.at("id")==selected_id){item=x;found=true;break;}if(!found)throw std::runtime_error("Selected image ID not found in manifest");}
  else {std::mt19937 rng(seed);std::uniform_int_distribution<size_t> pick(0,manifest["images"].size()-1);item=manifest["images"][pick(rng)];}
  std::string filename=item.at("file_name").get<std::string>();cv::Mat img=cv::imread((images/filename).string());if(img.empty())throw std::runtime_error("Cannot read image: "+filename);
  auto model=tflite::FlatBufferModel::BuildFromFile(model_path.string().c_str());if(!model)throw std::runtime_error("Cannot load TFLite model");tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
  std::unique_ptr<TfLiteDelegate,decltype(&TfLiteXNNPackDelegateDelete)> delegate(nullptr,TfLiteXNNPackDelegateDelete);std::unique_ptr<tflite::Interpreter> interpreter;
  check(tflite::InterpreterBuilder(*model,resolver)(&interpreter),"Cannot build interpreter");check(interpreter->SetNumThreads(threads),"Cannot set threads");
  if(backend=="xnnpack"){auto options=TfLiteXNNPackDelegateOptionsDefault();options.num_threads=threads;delegate.reset(TfLiteXNNPackDelegateCreate(&options));if(!delegate)throw std::runtime_error("Cannot create XNNPACK delegate");check(interpreter->ModifyGraphWithDelegate(delegate.get()),"XNNPACK delegate failed");}
  check(interpreter->AllocateTensors(),"Cannot allocate tensors");auto*in=interpreter->input_tensor(0);auto*out=interpreter->output_tensor(0);auto is=shape(in),os=shape(out);
  if(interpreter->inputs().size()!=1||interpreter->outputs().size()!=1||in->type!=kTfLiteFloat32||out->type!=kTfLiteFloat32)throw std::runtime_error("Expected one FP32 input and output");
  bool nhwc=is==std::vector<int>{1,640,640,3};if(!nhwc&&is!=std::vector<int>{1,3,640,640})throw std::runtime_error("Unsupported input shape");bool features_first=os==std::vector<int>{1,84,8400};if(!features_first&&os!=std::vector<int>{1,8400,84})throw std::runtime_error("Unsupported output shape");
  float ratio;int left,top;std::vector<double> pre,infer,post,total;json warmup_times=json::array();
  for(int i=0;i<warmup;i++){auto start=Clock::now();prepare(img,interpreter->typed_input_tensor<float>(0),nhwc,ratio,left,top);auto stage=Clock::now();check(interpreter->Invoke(),"Warmup inference failed");auto mid=Clock::now();auto dets=decode(interpreter->typed_output_tensor<float>(0),features_first,units=="normalized",ratio,left,top,img.cols,img.rows);(void)dets;auto end=Clock::now();if(i==0)warmup_times.push_back({{"preprocess_ms",std::chrono::duration<double,std::milli>(stage-start).count()},{"inference_ms",std::chrono::duration<double,std::milli>(mid-stage).count()},{"postprocess_ms",std::chrono::duration<double,std::milli>(end-mid).count()}});}
  std::cout<<"Benchmark image_id="<<item.at("id")<<" ("<<filename<<"), warmup="<<warmup<<", iterations="<<iterations<<"\n";
  for(int i=0;i<iterations;i++){auto whole=Clock::now();auto stage=Clock::now();prepare(img,interpreter->typed_input_tensor<float>(0),nhwc,ratio,left,top);auto infer_start=Clock::now();check(interpreter->Invoke(),"Inference failed");auto post_start=Clock::now();auto dets=decode(interpreter->typed_output_tensor<float>(0),features_first,units=="normalized",ratio,left,top,img.cols,img.rows);auto end=Clock::now();(void)dets;pre.push_back(std::chrono::duration<double,std::milli>(infer_start-stage).count());infer.push_back(std::chrono::duration<double,std::milli>(post_start-infer_start).count());post.push_back(std::chrono::duration<double,std::milli>(end-post_start).count());total.push_back(std::chrono::duration<double,std::milli>(end-whole).count());if((i+1)%100==0||i+1==iterations)std::cout<<"Measured "<<(i+1)<<"/"<<iterations<<" iterations\n";}
  json samples=json::array();for(int i=0;i<iterations;i++)samples.push_back({{"iteration",i+1},{"preprocess_ms",pre[i]},{"inference_ms",infer[i]},{"postprocess_ms",post[i]},{"total_ms",total[i]}});
  if(prefix.has_parent_path())fs::create_directories(prefix.parent_path());json summary={{"mode","single_image_benchmark"},{"runtime","TFLite C++ Interpreter API"},{"runtime_version",TFLITE_VERSION_STRING},{"backend",backend},{"model",model_path.string()},{"model_bytes",fs::file_size(model_path)},{"image_id",item.at("id")},{"image_file_name",filename},{"seed",actual_seed},{"seed_used",explicit_image?json(nullptr):json(seed)},{"warmup_iterations",warmup},{"measured_iterations",iterations},{"threads",threads},{"time_unit","ms"},{"std_definition","population (N divisor)"},{"input_shape",is},{"output_shape",os},{"box_units",units},{"precision",precision_metadata(*model->GetModel(),*interpreter->input_tensor(0),*interpreter->output_tensor(0))},{"statistics",{{"preprocess_ms",statsJson(pre)},{"inference_ms",statsJson(infer)},{"postprocess_ms",statsJson(post)},{"total_ms",statsJson(total)}}},{"warmup_sample",warmup_times},{"per_iteration",samples}};
  std::ofstream f(prefix.string()+"_benchmark.json");if(!f)throw std::runtime_error("Cannot write benchmark JSON");f<<summary.dump(2);return 0;
 }
 if(argc!=7&&argc!=8){std::cerr<<"Usage: yolo_tflite MODEL.tflite IMAGES MANIFEST OUTPUT_PREFIX THREADS normalized|pixels [xnnpack|builtin]\n";return 1;}
 fs::path images=argv[2],prefix=argv[4];int threads=std::stoi(argv[5]);
 std::string units=argv[6],backend=argc==8?argv[7]:"xnnpack";
 if(threads<1||(units!="normalized"&&units!="pixels")||(backend!="xnnpack"&&backend!="builtin"))throw std::runtime_error("Invalid threads, box units or backend");
 std::ifstream mf(argv[3]);if(!mf)throw std::runtime_error("Cannot open manifest");json manifest;mf>>manifest;
 if(manifest.at("images").empty()||manifest.at("category_ids").size()!=80)throw std::runtime_error("Invalid manifest");
 auto model=tflite::FlatBufferModel::BuildFromFile(argv[1]);if(!model)throw std::runtime_error("Cannot load TFLite model");
 tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates resolver;
 std::unique_ptr<TfLiteDelegate,decltype(&TfLiteXNNPackDelegateDelete)> delegate(nullptr,TfLiteXNNPackDelegateDelete);
 std::unique_ptr<tflite::Interpreter> interpreter;
 check(tflite::InterpreterBuilder(*model,resolver)(&interpreter),"Cannot build interpreter; check operator versions/custom ops");
 check(interpreter->SetNumThreads(threads),"Cannot set threads");
 if(backend=="xnnpack"){
  auto options=TfLiteXNNPackDelegateOptionsDefault();options.num_threads=threads;
  // Do not enable FORCE_FP16 or other reduced-precision flags.
  delegate.reset(TfLiteXNNPackDelegateCreate(&options));if(!delegate)throw std::runtime_error("Cannot create XNNPACK delegate");
  check(interpreter->ModifyGraphWithDelegate(delegate.get()),"XNNPACK delegate failed");
 }
 check(interpreter->AllocateTensors(),"Cannot allocate tensors");
 if(interpreter->inputs().size()!=1||interpreter->outputs().size()!=1)throw std::runtime_error("Expected one input and one raw YOLO output (NMS must be disabled)");
 auto*in=interpreter->input_tensor(0);auto*out=interpreter->output_tensor(0);
 auto is=shape(in),os=shape(out);
 std::cout<<"Input "<<json(is)<<" output "<<json(os)<<" runtime="<<TFLITE_VERSION_STRING<<" backend="<<backend<<"\n";
 if(in->type!=kTfLiteFloat32||out->type!=kTfLiteFloat32)throw std::runtime_error("This baseline requires FP32 input/output; quantized I/O not implemented");
 bool nhwc=is==std::vector<int>{1,640,640,3};
 if(!nhwc&&is!=std::vector<int>{1,3,640,640})throw std::runtime_error("Unsupported input shape");
 bool features_first=os==std::vector<int>{1,84,8400};
 if(!features_first&&os!=std::vector<int>{1,8400,84})throw std::runtime_error("Unsupported output: expected raw YOLOv8n 84 features x 8400 anchors");
 auto first=cv::imread((images/manifest["images"][0]["file_name"].get<std::string>()).string());
 if(first.empty())throw std::runtime_error("Cannot read first image");float r;int l,t;
 prepare(first,interpreter->typed_input_tensor<float>(0),nhwc,r,l,t);
 for(int i=0;i<5;i++)check(interpreter->Invoke(),"Warmup failed");
 json predictions=json::array(),timings=json::array();
 for(const auto&item:manifest["images"]){
  auto start=Clock::now();auto img=cv::imread((images/item["file_name"].get<std::string>()).string());
  if(img.empty())throw std::runtime_error("Cannot read image: "+item["file_name"].get<std::string>());
  double load_ms=elapsed(start);auto stage=Clock::now();
  prepare(img,interpreter->typed_input_tensor<float>(0),nhwc,r,l,t);double pre_ms=elapsed(stage);
  stage=Clock::now();check(interpreter->Invoke(),"Inference failed");double infer_ms=elapsed(stage);
  stage=Clock::now();auto dets=decode(interpreter->typed_output_tensor<float>(0),features_first,units=="normalized",r,l,t,img.cols,img.rows);
  double post_ms=elapsed(stage),total_ms=elapsed(start);
  for(auto&d:dets)predictions.push_back({{"image_id",item["id"]},{"category_id",manifest["category_ids"][d.cls]}, {"bbox",{d.x1,d.y1,d.x2-d.x1,d.y2-d.y1}},{"score",d.score}});
  timings.push_back({{"image_id",item["id"]},{"load_ms",load_ms},{"preprocess_ms",pre_ms},{"inference_ms",infer_ms},{"postprocess_ms",post_ms},{"total_ms",total_ms}});
  std::cout<<timings.size()<<"/"<<manifest["images"].size()<<" infer="<<infer_ms<<" total="<<total_ms<<" ms\n";
 }
 if(prefix.has_parent_path())fs::create_directories(prefix.parent_path());
 json means;for(auto key:{"load_ms","preprocess_ms","inference_ms","postprocess_ms","total_ms"}){double sum=0;for(auto&v:timings)sum+=v[key].get<double>();means[key]=sum/timings.size();}
 json summary={{"runtime","TFLite C++ Interpreter API"},{"runtime_version",TFLITE_VERSION_STRING},{"backend",backend},{"model",argv[1]},{"model_bytes",fs::file_size(argv[1])},{"input_shape",is},{"output_shape",os},{"box_units",units},{"precision",precision_metadata(*model->GetModel(),*interpreter->input_tensor(0),*interpreter->output_tensor(0))},{"threads",threads},{"imgsz",640},{"batch",1},{"warmup",5},{"conf",0.001},{"nms_iou",0.7},{"max_det",100},{"means",means},{"per_image",timings},{"manifest",manifest}};
 auto write=[](const std::string&p,const json&j){std::ofstream f(p);if(!f)throw std::runtime_error("Cannot write "+p);f<<j.dump(2);};
 write(prefix.string()+"_predictions.json",predictions);write(prefix.string()+"_timing.json",summary);std::cout<<means.dump(2)<<"\n";
 return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
