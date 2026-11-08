#include <net.h>
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
#include <stdexcept>
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
static ncnn::Mat prepare(const cv::Mat&img,float&r,int&left,int&top){
 r=std::min(640.f/img.cols,640.f/img.rows);
 int w=int(std::nearbyint(img.cols*r)),h=int(std::nearbyint(img.rows*r));
 left=int(std::nearbyint((640-w)/2.0-0.1)); top=int(std::nearbyint((640-h)/2.0-0.1));
 cv::Mat resized,padded;
 cv::resize(img,resized,cv::Size(w,h),0,0,cv::INTER_LINEAR);
 cv::copyMakeBorder(resized,padded,top,640-h-top,left,640-w-left,cv::BORDER_CONSTANT,cv::Scalar(114,114,114));
 auto input=ncnn::Mat::from_pixels(padded.data,ncnn::Mat::PIXEL_BGR2RGB,640,640);
 const float norm[3]={1.f/255,1.f/255,1.f/255}; input.substract_mean_normalize(nullptr,norm); return input;
}
static ncnn::Mat infer(ncnn::Net&net,const ncnn::Mat&input){
 auto ex=net.create_extractor(); ncnn::Mat output;
 if(ex.input("in0",input)!=0||ex.extract("out0",output)!=0)throw std::runtime_error("NCNN input/extract failed");
 return output;
}
static std::vector<Detection> decode(const ncnn::Mat&out,float ratio,int left,int top,int width,int height){
 if(out.dims!=2||out.elempack!=1||out.elembits()!=32||!((out.w==8400&&out.h==84)||(out.w==84&&out.h==8400)))
  throw std::runtime_error("Unexpected out0 shape/type: expected unpacked FP32 84x8400 or 8400x84");
 auto at=[&](int feature,int anchor){return out.w==8400?out.row(feature)[anchor]:out.row(anchor)[feature];};
 std::vector<Detection> candidates;
 for(int a=0;a<8400;a++){
  int cls=0;float score=at(4,a);
  for(int c=1;c<80;c++)if(at(c+4,a)>score){score=at(c+4,a);cls=c;}
  if(score<0.001f)continue;
  float cx=at(0,a),cy=at(1,a),w=at(2,a),h=at(3,a);
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
int main(int argc,char**argv){try{
 if(argc<5){std::cerr<<"Usage: yolo_ncnn MODEL_DIR IMAGE_DIR MANIFEST_JSON OUTPUT_PREFIX [threads=4]\n";return 1;}
 fs::path model=argv[1],images=argv[2],prefix=argv[4];int threads=argc>5?std::stoi(argv[5]):4;
 if(threads<1)throw std::runtime_error("threads must be positive");
 std::ifstream mf(argv[3]);if(!mf)throw std::runtime_error("Cannot read manifest");json manifest;mf>>manifest;
 if(manifest.at("images").empty()||manifest.at("category_ids").size()!=80)throw std::runtime_error("Invalid manifest");
 ncnn::Net net;net.opt.num_threads=threads;net.opt.use_vulkan_compute=false;
 net.opt.use_fp16_packed=false;net.opt.use_fp16_storage=false;net.opt.use_fp16_arithmetic=false;
 net.opt.use_bf16_storage=false;net.opt.use_int8_inference=false;
 if(net.load_param((model/"model.ncnn.param").string().c_str())||net.load_model((model/"model.ncnn.bin").string().c_str()))throw std::runtime_error("Cannot load NCNN model");
 auto first=cv::imread((images/manifest["images"][0]["file_name"].get<std::string>()).string());
 if(first.empty())throw std::runtime_error("Cannot read first image");float r;int l,t;auto warm=prepare(first,r,l,t);
 for(int i=0;i<5;i++)infer(net,warm);
 json predictions=json::array(),timings=json::array();bool shape_printed=false;
 for(const auto&item:manifest.at("images")){
  auto start=Clock::now();auto img=cv::imread((images/item.at("file_name").get<std::string>()).string());
  if(img.empty())throw std::runtime_error("Cannot read image: "+item.at("file_name").get<std::string>());
  double load_ms=elapsed(start);auto stage=Clock::now();auto input=prepare(img,r,l,t);double pre_ms=elapsed(stage);
  stage=Clock::now();auto out=infer(net,input);double infer_ms=elapsed(stage);
  stage=Clock::now();auto dets=decode(out,r,l,t,img.cols,img.rows);double post_ms=elapsed(stage);double total_ms=elapsed(start);
  if(!shape_printed){std::cout<<"out0: dims="<<out.dims<<" w="<<out.w<<" h="<<out.h<<" c="<<out.c<<"\n";shape_printed=true;}
  for(auto&d:dets)predictions.push_back({{"image_id",item.at("id")},{"category_id",manifest["category_ids"][d.cls]},
   {"bbox",{d.x1,d.y1,d.x2-d.x1,d.y2-d.y1}},{"score",d.score}});
  timings.push_back({{"image_id",item.at("id")},{"load_ms",load_ms},{"preprocess_ms",pre_ms},{"inference_ms",infer_ms},{"postprocess_ms",post_ms},{"total_ms",total_ms}});
  std::cout<<timings.size()<<"/"<<manifest["images"].size()<<" inference="<<infer_ms<<" ms total="<<total_ms<<" ms\n";
 }
 if(prefix.has_parent_path())fs::create_directories(prefix.parent_path());
 json means;for(auto key:{"load_ms","preprocess_ms","inference_ms","postprocess_ms","total_ms"}){double sum=0;for(auto&v:timings)sum+=v[key].get<double>();means[key]=sum/timings.size();}
 json summary={{"runtime","ncnn"},{"ncnn_build_tag","20250503"},{"precision","FP32 weights and compute"},{"threads",threads},{"imgsz",640},{"batch",1},{"warmup",5},{"conf",0.001},{"nms_iou",0.7},{"max_det",100},{"means",means},{"per_image",timings},{"manifest",manifest}};
 auto write=[](const std::string&p,const json&j){std::ofstream f(p);if(!f)throw std::runtime_error("Cannot write "+p);f<<j.dump(2);};
 write(prefix.string()+"_predictions.json",predictions);write(prefix.string()+"_timing.json",summary);
 std::cout<<means.dump(2)<<"\n";return 0;
 }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 2;}}
