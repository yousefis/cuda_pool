#pragma once

#include <gpu.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "yololayer.h"

#include <image.h>

using namespace nvinfer1;

//TODO don't use _ in class / function names
//TODO make base class YoloV5sModel, subclass CarDetectorModel that implements 
// methods like "isCarDetected", that internally calls YoloV5sModel::isDetected(...) or similar

class YoloV5s_Model : public ILogger
{
public:
	YoloV5s_Model(const char* filename);
	virtual ~YoloV5s_Model();
	
	Image<float>* inputFrame = nullptr;
	Yolo::Output *output = nullptr; 
	
	//TODO: These needs to be taken from the settings. Needs to change this.
	struct { float confidence = 0.4, nms = 0.4, area = 0.2; } threshold;
	bool  isDetected = false;
	int   detectionCount = 0;
	float detectionClass = 0;

	virtual void log(Severity level, const char* msg) override
	{
		printf("[L%d] %s\n", (int) level, msg);
	}

	void infer(cudaStream_t);
	
	template <typename T>
	void renderBoxes(Image<T>* dst, cudaStream_t stream);

private:	
	void load(const char* filename);
	void setup();
	ICudaEngine* _engine = nullptr;
	IRuntime*    _runtime = nullptr;
	IExecutionContext* _context = nullptr;
};
