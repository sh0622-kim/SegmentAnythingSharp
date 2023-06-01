using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace SegmentAnythingSharp
{
    public class SamRunner
    {
        private InferenceSession _sessionPreprocess, _sessionSegmentAnything;
        private int[] _inputShape;
        private bool _modelLoaded = false;
        private DenseTensor<float>? _imageEmbeddings;

        public SamRunner(string preprocessModelPath, string segmentAnythingModelPath)
        {
            var sessionPrePath = Path.GetFullPath(preprocessModelPath);
            var sessionSamPath = Path.GetFullPath(segmentAnythingModelPath);

            _sessionPreprocess = new InferenceSession(sessionPrePath);

            _sessionSegmentAnything = new InferenceSession(sessionSamPath);

            _inputShape = _sessionPreprocess.InputMetadata["input"].Dimensions;

            _modelLoaded = true;
        }

        public Size GetInputSize()
        {
            if (!_modelLoaded)
            {
                return new Size(0, 0);
            }

            return new Size(_inputShape[3], _inputShape[2]);
        }

        public bool LoadImage(Mat image)
        {
            var inputTensorValues = new byte[_inputShape[0] * _inputShape[1] * _inputShape[2] * _inputShape[3]];

            for (int i = 0; i < _inputShape[2]; i++)
            {
                for (int j = 0; j < _inputShape[3]; j++)
                {
                    var pixel = image.Get<Vec3b>(i, j);
                    inputTensorValues[(i * _inputShape[3]) + j] = pixel[2];
                    inputTensorValues[(_inputShape[2] * _inputShape[3]) + (i * _inputShape[3]) + j] = pixel[1];
                    inputTensorValues[(2 * _inputShape[2] * _inputShape[3]) + (i * _inputShape[3]) + j] = pixel[0];
                }
            }

            var inputTensor = new DenseTensor<byte>(inputTensorValues, _inputShape.Select(d => d).ToArray());

            string[] inputNamesPre = { "input" };

            var container = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor)
            };

            List<DisposableNamedOnnxValue> results = (List<DisposableNamedOnnxValue>)_sessionPreprocess.Run(container);

            _imageEmbeddings = (DenseTensor<float>)results[0].Value;

            return true;
        }

        public Mat GetMask(Point point)
        {
            const int maskInputSize = 256 * 256;
            float[] inputPointValues = { point.X, point.Y };
            float[] inputLabelValues = { 1 };
            float[] maskInputValues = new float[maskInputSize];
            float[] hasMaskValues = { 0 };
            float[] origImSizeValues = { _inputShape[2], _inputShape[3] };
            Array.Clear(maskInputValues, 0, maskInputValues.Length);

            int numPoints = 1;
            var inputPointShape = new[] { 1, numPoints, 2 };
            var pointLabelsShape = new[] { 1, numPoints };
            var maskInputShape = new[] { 1, 1, 256, 256 };
            var hasMaskInputShape = new[] { 1 };
            var origImSizeShape = new[] { 2 };

            var pointCoords = new DenseTensor<float>(inputPointValues, inputPointShape);
            var pointLabels = new DenseTensor<float>(inputLabelValues, pointLabelsShape);
            var maskInput = new DenseTensor<float>(maskInputValues, maskInputShape);
            var hasMaskInput = new DenseTensor<float>(hasMaskValues, hasMaskInputShape);
            var imageSize = new DenseTensor<float>(origImSizeValues, origImSizeShape);

            var container = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("image_embeddings", _imageEmbeddings),
                NamedOnnxValue.CreateFromTensor("point_coords", pointCoords),
                NamedOnnxValue.CreateFromTensor("point_labels", pointLabels),
                NamedOnnxValue.CreateFromTensor("mask_input", maskInput),
                NamedOnnxValue.CreateFromTensor("has_mask_input", hasMaskInput),
                NamedOnnxValue.CreateFromTensor("orig_im_size", imageSize)
            };

            List<DisposableNamedOnnxValue> results = (List<DisposableNamedOnnxValue>)_sessionSegmentAnything.Run(container);

            var outputMasksValues = results[0].AsTensor<float>().ToArray();
            var outputMaskSam = new Mat(_inputShape[2], _inputShape[3], MatType.CV_8UC1);
            for (int i = 0; i < outputMaskSam.Rows; i++)
            {
                for (int j = 0; j < outputMaskSam.Cols; j++)
                {
                    outputMaskSam.Set(i, j, outputMasksValues[(i * outputMaskSam.Cols) + j] > 0 ? (byte)255 : (byte)0);
                }
            }
            return outputMaskSam;
        }
    }
}
