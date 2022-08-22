#include "yolov5s.h"

__global__
static void nms_gpu(Yolo::Output* out, float nms_thresh )
{
	auto idx = blockIdx * blockDim + threadIdx;
	if (idx.x >= out->count || idx.y >= out->count) return;
	if (idx.x >= idx.y) return;

	auto& bb0 = out->boxes[idx.x];
	auto& bb1 = out->boxes[idx.y];
	if (bb0.class_id != bb1.class_id) return;

	auto area0 = bb0.width * bb0.height;
	auto area1 = bb1.width * bb1.height;
	auto x_min = fmaxf(fmaf(bb0.width,  -0.5f, bb0.cx), fmaf(bb1.width,  -0.5f, bb1.cx));
	auto x_max = fminf(fmaf(bb0.width,  +0.5f, bb0.cx), fmaf(bb1.width,  +0.5f, bb1.cx));
	auto y_min = fmaxf(fmaf(bb0.height, -0.5f, bb0.cy), fmaf(bb1.height, -0.5f, bb1.cy));
	auto y_max = fminf(fmaf(bb0.height, +0.5f, bb0.cy), fmaf(bb1.height, +0.5f, bb1.cy));

	auto intersect_w = fmaxf(0.0f, x_max - x_min);
	auto intersect_h = fmaxf(0.0f, y_max - y_min);
	auto intersect_area = intersect_w * intersect_h;
	auto iou = (intersect_area) / (area0 + area1 - intersect_area);

	if (iou >= nms_thresh)
	{
		if (bb1.conf > bb0.conf) bb0.conf = 0;
		else bb1.conf = 0;
	}
}
YoloV5s_Model::YoloV5s_Model(const char* filename)
{
	load(filename);
	setup();
}

YoloV5s_Model::~YoloV5s_Model()
{
	if (inputFrame) delete inputFrame;;
	if (output) cudaFreeHost(output);

	if (_context) _context->destroy();
	if (_engine)  _engine->destroy();
	if (_runtime) _runtime->destroy();
}

void YoloV5s_Model::load(const char* filename)
{
	std::cout<<"Loading the model from file : "<<filename<<std::endl;
	std::ifstream file(filename, std::ios::binary);
	if (!file.good()) {
		std::cerr << "read " << filename << " error!" << std::endl;
                return;
        }
    
	char *trtYoloV5s_ModelStream = nullptr;
	size_t size = 0;
	file.seekg(0, file.end);
	size = file.tellg();
	file.seekg(0, file.beg);
	trtYoloV5s_ModelStream = new char[size];
	assert(trtYoloV5s_ModelStream);
   	file.read(trtYoloV5s_ModelStream, size);
	file.close();
	printf("Engine file read successfully.\nCreating the engine context. ");

   	
	_runtime = createInferRuntime(*this);
	assert(_runtime != nullptr);

	_engine = _runtime->deserializeCudaEngine(trtYoloV5s_ModelStream, size);
	assert(_engine != nullptr);

	_context = _engine->createExecutionContext();
	assert(_context != nullptr);

	delete[] trtYoloV5s_ModelStream;
	assert(_engine->getNbBindings() == 2);
	printf("Successfully deserialised the engine.\n");

}

void YoloV5s_Model::setup()
{
	int rc;
	
	printf("Creating YoloV5s_Model frames\n");
	assert(640 == Yolo::INPUT_W);
	assert(640 == Yolo::INPUT_H);
	inputFrame = new Image<float>(3, Yolo::INPUT_W, Yolo::INPUT_H);
	assert(inputFrame->channel[0]->pitch == inputFrame->channel[0]->width * sizeof(float));

	rc = cudaHostAlloc(&output, sizeof(Yolo::Output), cudaHostAllocMapped);
	if (cudaSuccess != rc) throw "Unable to allocate boxes frame device memory";
}

template <typename T>
__global__
static void f_render_boxes(Channel<T>* y, Channel<T>* u, Channel<T>* v, const Yolo::Output* out)
{
	auto stride = blockDim * gridDim;
	auto offset = blockIdx * blockDim + threadIdx;
	
	for (auto row = offset.y; row < u->height; row+=stride.y)
	{
		auto dy = y->row(row);
		auto du = u->row(row);
		auto dv = v->row(row);
		for (auto col = offset.x; col < u->width; col+=stride.x)
		{
			for (int i = 0; i<out->count; i++)
			{
				auto& bb = out->boxes[i];
				if ( 0 != bb.conf
				  && 2 * fabsf(bb.cx - col) < bb.width
				  && 2 * fabsf(bb.cy - row) < bb.height)
					dy[col] = fmaf(dy[col], 0.7, 0.3);
			}
		}
	}
}

template <typename T>
void YoloV5s_Model::renderBoxes(Image<T>* dst, cudaStream_t stream)
{
	dim3 blockSize = { 16, 16 };
	dim3 gridSize = { 32, 32 };
	f_render_boxes <T> <<< gridSize, blockSize, 0, stream  >>> (
			dst->channel[0], 
			dst->channel[1], 
			dst->channel[2], 
			output);

}

template void YoloV5s_Model::renderBoxes(Image<uint8_t>*,  cudaStream_t);
template void YoloV5s_Model::renderBoxes(Image<uint16_t>*, cudaStream_t);
template void YoloV5s_Model::renderBoxes(Image<float>*,    cudaStream_t);

void YoloV5s_Model::infer(cudaStream_t stream)
{
	// TODO enqueue in batches of 2. 
	assert(output);
	assert(inputFrame);
	assert(inputFrame->channel[0]);
	assert(inputFrame->channel[0]->data);

	void* bindings[] = { inputFrame->channel[0]->data, output };
	bool result = _context->enqueue(1, bindings, stream, nullptr);
	if (!result) 
	{
		throw "Unable to enqueue yolo inference";
	}


	dim3 blockSize = {16, 16};
	dim3 gridSize =  { 
		(Yolo::MAX_OUTPUT_BBOX_COUNT + blockSize.x - 1) / blockSize.x,
		(Yolo::MAX_OUTPUT_BBOX_COUNT + blockSize.y - 1) / blockSize.y };

	nms_gpu <<< gridSize, blockSize, 0, stream >>> (
			output,
			threshold.nms);

	cudaStreamSynchronize(stream);
	
	int dc = 0;
	float totalArea = Yolo::INPUT_H * Yolo::INPUT_W;
	for (int i=0; i<(int) output->count; i++)
	{
		auto& bb = output->boxes[i];

		if ( bb.conf >= threshold.confidence 
		  && bb.width * bb.height / totalArea >= threshold.area) dc++;
		  //&& detectionClass == bb.class_id) dc++; // Shouldn't remove the person detections as it's needed for tracking.
		else bb.conf = 0;
	}
	detectionCount = dc;
	isDetected = detectionCount > 0;
}
