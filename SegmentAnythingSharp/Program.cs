using ConsoleApp1;
using OpenCvSharp;

// sam model path
string preModelPath = ".\\models\\sam_preprocess.onnx";
string samModelPath = ".\\models\\sam_vit_h_4b8939.onnx";

// image path
string inputImagePath = ".\\images\\truck.jpg";
string resizeImagePath = ".\\results\\resize.jpg";
string maskImagePath = ".\\results\\mask.jpg";
string dstImagePath = ".\\results\\dst.jpg";

// point
Point point = new Point(300, 220);

SamRunner samRunner = new SamRunner(preModelPath, samModelPath);

Mat src = Cv2.ImRead(inputImagePath);
Mat dst = Mat.Zeros(samRunner.GetInputSize(), MatType.CV_8UC3);
Cv2.Resize(src, src, samRunner.GetInputSize());
src.SaveImage(resizeImagePath);

samRunner.LoadImage(src);
Mat mask = samRunner.GetMask(point);
mask.SaveImage(maskImagePath);

for (int i = 0; i < src.Rows; i++)
{
    for (int j = 0; j < src.Cols; j++)
    {
        var bFront = mask.At<byte>(i, j) > 0;
        float factor = bFront ? 1.0f : 0.2f;
        dst.Set(i, j, src.At<Vec3b>(i, j) * factor);
    }
}

Cv2.DrawMarker(dst, point, Scalar.Lime, MarkerTypes.Cross, 20, 2);

dst.SaveImage(dstImagePath);