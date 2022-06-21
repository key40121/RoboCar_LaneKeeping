#include <iostream>
#include <tuple>
#include <cmath> // abs() for float and double, asin()
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//ver2
//double ymm_per_pix = 200 / 240;
//double xmm_per_pix = 375 / 160;


//ver3
double ymm_per_pix = 300 / 240;
double xmm_per_pix = 365 / 220;

std::vector<cv::Point2f> SlidingWindow(cv::Mat image, cv::Rect);
std::tuple<double, double> Polynomial(std::vector<cv::Point2f>);
std::tuple<cv::Mat, cv::Mat> CalibMatrix();

cv::Mat ImageCalibration(cv::Mat img);
cv::Mat ImageBirdsEyeProcess(cv::Mat img);
cv::Mat ImageProcessing(cv::Mat img);

// test 
void test_1();
void test_2();
void test_for_robocar();

int main() {
	// test_1();
	test_2();
}


std::vector<cv::Point2f> SlidingWindow(cv::Mat image, cv::Rect window) {
	std::vector<cv::Point2f> points;
	const cv::Size imgSize = image.size();
	bool shouldBreak = false;

	//test 
	//cv::Point2f start = cv::Point2f(77, 240);
	//points.push_back(start);

	while (true)
	{
		float currentX = window.x + window.width * 0.5f;

		// Extract region of interest.
		cv::Mat roi;
		roi = cv::Mat(image, window);

		std::vector<cv::Point2f> locations;

		// Get all non-black pixels. All pixels are white in our case.
		cv::findNonZero(roi, locations);
		float avgX = 0.0f;

		// Calculate average X position
		for (int i = 0; i < locations.size(); ++i)
		{
			float x = locations[i].x;
			avgX += window.x + x;
		}

		// sankou enzanshi
		avgX = locations.empty() ? currentX : avgX / locations.size();

		cv::Point point(avgX, window.y + window.height * 0.5f);
		points.push_back(point);

		// Move the window up
		window.y -= window.height;

		// For the uppermost position
		if (window.y < 0)
		{
			window.y = 0;
			shouldBreak = true;
		}

		// Move the x position
		window.x += (point.x - currentX);

		// Make sure the window dosent overflow, we get an error if we try to fet data outside the mattrix
		if (window.x < 0)
		{
			window.x = 0;
		}

		if (window.x + window.width >= imgSize.width)
		{
			window.x = imgSize.width - window.width - 1;
		}

		if (shouldBreak)
		{
			break;
		}
	}
	return points;
}

std::tuple<double, double> Polynomial(std::vector<cv::Point2f> pts)
{
	int i, j, k, n, N;


	// The number of the point

	N = 8;
	double x[8];
	double y[8];

	// N = 8;
	// double x[8];
	// double y[8];

	for (int i = 0; i < pts.size(); i++)
	{
		x[i] = pts[i].x * xmm_per_pix;
		y[i] = pts[i].y * ymm_per_pix;
	}


	//std::cout << "\nEnter the x-axis values:\n";                //Input x-values
	//for (i = 0; i < N; i++)
	//	std::cin >> x[i];
	//std::cout << "\nEnter the y-axis values:\n";                //Input y-values
	//for (i = 0; i < N; i++)
	//	std::cin >> y[i];

	// n is the degree of Polynomial 
	n = 2;

	// 

	double X[5];                        //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
	for (i = 0; i < 2 * n + 1; i++)
	{
		X[i] = 0;
		for (j = 0; j < N; j++)
			X[i] = X[i] + pow(x[j], i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
	}
	double B[3][4], a[3];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
	for (i = 0; i <= n; i++)
		for (j = 0; j <= n; j++)
			B[i][j] = X[i + j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
	double Y[3];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	for (i = 0; i < n + 1; i++)
	{
		Y[i] = 0;
		for (j = 0; j < N; j++)
			Y[i] = Y[i] + pow(x[j], i) * y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
	}
	for (i = 0; i <= n; i++)
		B[i][n + 1] = Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
	n = n + 1;                //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations
	std::cout << "\nThe Normal(Augmented Matrix) is as follows:\n";
	for (i = 0; i < n; i++)            //print the Normal-augmented matrix
	{
		for (j = 0; j <= n; j++)
			std::cout << B[i][j] << std::setw(16);
		std::cout << "\n";
	}
	for (i = 0; i < n; i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
		for (k = i + 1; k < n; k++)
			if (B[i][i] < B[k][i])
				for (j = 0; j <= n; j++)
				{
					double temp = B[i][j];
					B[i][j] = B[k][j];
					B[k][j] = temp;
				}

	for (i = 0; i < n - 1; i++)            //loop to perform the gauss elimination
		for (k = i + 1; k < n; k++)
		{
			double t = B[k][i] / B[i][i];
			for (j = 0; j <= n; j++)
				B[k][j] = B[k][j] - t * B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
		}
	for (i = n - 1; i >= 0; i--)                //back-substitution
	{                        //x is an array whose values correspond to the values of x,y,z..
		a[i] = B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
		for (j = 0; j < n; j++)
			if (j != i)            //then subtract all the lhs values except the coefficient of the variable whose value                                   is being calculated
				a[i] = a[i] - B[i][j] * a[j];
		a[i] = a[i] / B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
	}
	std::cout << "\nThe values of the coefficients are as follows:\n";
	for (i = 0; i < n; i++)
		std::cout << "x^" << i << "=" << a[i] << std::endl;            // Print the values of x^0,x^1,x^2,x^3,....    
	std::cout << "\nHence the fitted Polynomial is given by:\ny=";
	for (i = 0; i < n; i++)
		std::cout << " + (" << a[i] << ")" << "x^" << i;
	std::cout << "\n";


	return std::forward_as_tuple(a[2], a[1]);
}

std::tuple<cv::Mat, cv::Mat> CalibMatrix()
{
	// call calibration using distortion matrix
	cv::Mat camera_matrix;
	cv::Mat distortion_coeff;
	
	cv::FileStorage fs("calibration_matrix.xml", cv::FileStorage::READ);
	fs["intrinsic"] >> camera_matrix;
	fs["distortion"] >> distortion_coeff;
	fs.release();

	return std::forward_as_tuple(camera_matrix, distortion_coeff);
}

cv::Mat ImageCalibration(cv::Mat img)
{
	// calibration using distortion matrix
	cv::Mat camera_matrix;
	cv::Mat distortion_coeff;

	cv::Mat undistort_img;

	std::tie(camera_matrix, distortion_coeff) = CalibMatrix();

	cv::undistort(img, undistort_img, camera_matrix, distortion_coeff);

	return undistort_img;
}

cv::Mat ImageBirdsEyeProcess(cv::Mat img)
{
	// ver2
	//cv::Point2f srcVertices[4];
	//srcVertices[0] = cv::Point(80,79);
	//srcVertices[1] = cv::Point(233, 79);
	//srcVertices[2] = cv::Point(320, 137);
	//srcVertices[3] = cv::Point(0, 137);

	//cv::Point2f dstVertices[4];
	//dstVertices[0] = cv::Point(80, 0);
	//dstVertices[1] = cv::Point(240, 0);
	//dstVertices[2] = cv::Point(240, 240);
	//dstVertices[3] = cv::Point(80, 240);


	//// rectangles ver3
	cv::Point2f srcVertices[4];
	srcVertices[0] = cv::Point(100, 65);
	srcVertices[1] = cv::Point(214, 65);
	srcVertices[2] = cv::Point(320, 137);
	srcVertices[3] = cv::Point(0, 137);

	cv::Point2f dstVertices[4];
	dstVertices[0] = cv::Point(50, 0);
	dstVertices[1] = cv::Point(270, 0);
	dstVertices[2] = cv::Point(270, 240);
	dstVertices[3] = cv::Point(50, 240);

	// Prepare matrix for transform and get the warped image
	cv::Mat perspectiveMatrix = cv::getPerspectiveTransform(srcVertices, dstVertices);
	cv::Mat dst(240, 320, CV_8UC3); // Destination for warped image

	// For transforming back into original space
	cv::Mat invertedPerspectiveMatrix;
	cv::invert(perspectiveMatrix, invertedPerspectiveMatrix);

	cv:warpPerspective(img, dst, perspectiveMatrix, dst.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT);

	return dst;
}

cv::Mat ImageProcessing(cv::Mat img)
{
	cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
	
	// thresholding
	cv::Mat processed;
	const int THRESHOLD_VAL = 150;
	cv::threshold(img, processed, THRESHOLD_VAL, 255, cv::THRESH_BINARY);

	return processed;
}

double SteerAngle(double radius_of_curvature)
{
	double steer_angle;
	const double ROBOCAR_WIDTH = 429;
	const double PI = 3.141592;

	if (radius_of_curvature > 5000)
	{
		steer_angle = 0;
	}
	else
	{
		steer_angle = std::asin(ROBOCAR_WIDTH / radius_of_curvature) * 360 / 2 / PI;
	}
	
	return steer_angle;
}

void test_1()
{
	cv::Mat img;
	cv::Mat processed;
	cv::Mat processed_2;

	img = cv::imread("image/test_1.jpg");

	// calibration
	processed = ImageCalibration(img);

	// perspectice matrix transformation
	processed_2 = ImageBirdsEyeProcess(processed);

	// sliding window algorithm
	std::vector<cv::Point2f> pts = SlidingWindow(processed_2, cv::Rect(20, 210, 60, 30));

	// Polynomial fitting
	double poly_co, lin_co;
	std::tie(poly_co, lin_co) = Polynomial(pts);

	// Radius of curvature
	double radius_of_curvature;
	radius_of_curvature = std::pow((1 + std::pow(2 * poly_co * pts[2].x * xmm_per_pix + lin_co, 2)), 1.5) / abs(2 * poly_co);

	std::cout << radius_of_curvature << std::endl;


	cv::imshow("test", processed_2);
	cv::waitKey(0);
}

void test_2()
{
	cv::Mat img;
	cv::Mat processed;

	img = cv::imread("image/test_4.jpg");

	// calibration
	processed = ImageCalibration(img);

	// Perspective
	processed = ImageBirdsEyeProcess(processed);

	// Image processing
	processed = ImageProcessing(processed);

	// sliding window algorithm
	// ver2
	//std::vector<cv::Point2f> pts = SlidingWindow(processed, cv::Rect(80, 210, 60, 30));
	// ver3
	std::vector<cv::Point2f> pts = SlidingWindow(processed, cv::Rect(50, 210, 60, 30));

	// ver3 sliding window 16
	//std::vector<cv::Point2f> pts = SlidingWindow(processed, cv::Rect(50, 225, 60, 15));

	// Polynomial fitting
	double poly_co, lin_co;
	std::tie(poly_co, lin_co) = Polynomial(pts);

	// Radius of curvature
	double radius_of_curvature;
	radius_of_curvature = std::pow((1 + std::pow(2 * poly_co * pts[3].x * xmm_per_pix + lin_co, 2)), 1.5) / abs(2 * poly_co);

	std::cout << radius_of_curvature << std::endl;

	// have to think about the width of the car and the lane.
	radius_of_curvature = radius_of_curvature - 182.5;

	// steering angle
	double steering_angle;
	steering_angle = SteerAngle(radius_of_curvature);

	std::cout << steering_angle << std::endl;


	// show test image
	cv::imshow("test", processed);
	cv::waitKey(0);
}

void test_for_robocar()
{
	cv::Mat img;
	cv::Mat processed;

	img = cv::imread("image/test_4.jpg");

	// calibration
	processed = ImageCalibration(img);

	// Perspective
	processed = ImageBirdsEyeProcess(processed);

	// Image processing
	processed = ImageProcessing(processed);

	// sliding window algorithm
	// ver3
	std::vector<cv::Point2f> pts = SlidingWindow(processed, cv::Rect(50, 210, 60, 30));

	// Polynomial fitting
	double poly_co, lin_co;
	std::tie(poly_co, lin_co) = Polynomial(pts);

	// Radius of curvature
	double radius_of_curvature;
	radius_of_curvature = std::pow((1 + std::pow(2 * poly_co * pts[3].x * xmm_per_pix + lin_co, 2)), 1.5) / abs(2 * poly_co);

	std::cout << radius_of_curvature << std::endl;

	// have to think about the width of the car and the lane.
	radius_of_curvature = radius_of_curvature - 182.5;

	// steering angle
	double steering_angle;
	steering_angle = SteerAngle(radius_of_curvature);

	std::cout << steering_angle << std::endl;


	// show test image
	cv::imshow("test", processed);
	cv::waitKey(0);
}