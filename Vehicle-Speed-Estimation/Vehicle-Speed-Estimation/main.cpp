#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <string>

using namespace cv;
using namespace std;

Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

double distanceBetweenPoints(Point2f point1, Point point2);
double distanceBetweenPoints(Point2f point1, Point2f point2);
Point3f findWorldPoint(Point2f imagePoint, Mat cameraMatrix, Mat rotationMatrix, Mat translationVector);

class Blob
{
public:
	vector<Point> contour;
	Rect Bounding_Rectangle;
	RotatedRect Rotated_Rect;
	double area, width, height, diagonalSize, aspectRatio, averageFlowDistanceX, averageFlowDistanceY, angleOfMotion;
	Point bottomLeftCorner, topRightCorner, bottomRightCorner;
	Point2d center;
	vector<Point3d> groundPlaneFlowTails, groundPlaneFlowHeads;

	Mat mask;

	vector<Point2f> featurePoints;
	vector<Point2f> flowPoints;


	void findFeatures(Mat);
	void findFlow(Mat, Mat);


	Blob(vector<Point>);
	Blob();
	~Blob();
};
Blob::Blob() {}
Blob::~Blob() {}

Blob::Blob(vector<Point> contour)
{
	this->contour = contour;
	this->Bounding_Rectangle = boundingRect(this->contour);
	this->Rotated_Rect = minAreaRect(this->contour);
	this->area = this->Bounding_Rectangle.area();
	this->width = this->Bounding_Rectangle.width;
	this->height = this->Bounding_Rectangle.height;
	this->aspectRatio = (float)this->Bounding_Rectangle.width / (float)this->Bounding_Rectangle.height;
	this->diagonalSize = sqrt(pow(this->Bounding_Rectangle.width, 2) + pow(this->Bounding_Rectangle.height, 2));
	this->topRightCorner = Point(this->Bounding_Rectangle.x, this->Bounding_Rectangle.y);
	this->center.x = this->Bounding_Rectangle.x + (this->Bounding_Rectangle.width / 2);
	this->center.y = this->Bounding_Rectangle.y + (this->Bounding_Rectangle.height / 2);

	this->bottomLeftCorner.x = this->Bounding_Rectangle.x;
	this->bottomLeftCorner.y = this->Bounding_Rectangle.y + this->Bounding_Rectangle.height;

	this->bottomRightCorner.x = this->Bounding_Rectangle.x + this->Bounding_Rectangle.width;
	this->bottomRightCorner.y = this->Bounding_Rectangle.y + this->Bounding_Rectangle.height;

}

void Blob::findFeatures(Mat currentFrame)
{
	vector<Point2f> featurePoints;
	vector<vector<Point> > contours;
	contours.push_back(this->contour);
	Mat mask(currentFrame.size(), CV_8UC1, Scalar(0, 0, 0));
	drawContours(mask, contours, -1, Scalar(255, 255, 255), -1);
	this->mask = mask;

	goodFeaturesToTrack(currentFrame, featurePoints, 100, 0.01, 5, mask);
	cornerSubPix(currentFrame, featurePoints, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	this->featurePoints = featurePoints;
	for (int i = 0; i < this->featurePoints.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(featurePoints[i]), cameraMatrix, rotationMatrix, translationVector);
		this->groundPlaneFlowTails.push_back(point);
	}

}

void Blob::findFlow(Mat currentFrame, Mat nextFrame)
{

	int win_size = 10; vector<uchar> status; vector<float> error;
	calcOpticalFlowPyrLK(currentFrame, nextFrame, this->featurePoints, flowPoints, status, error, Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));
	this->flowPoints = flowPoints;

	for (int i = 0; i < featurePoints.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowPoints[i]), cameraMatrix, rotationMatrix, translationVector);
		this->groundPlaneFlowHeads.push_back(point);
	}


	double totalDx = 0, totalDy = 0;

	for (int i = 0; i < this->groundPlaneFlowTails.size(); i++)
	{

		double dx = groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x;
		double dy = groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y;
		totalDx = totalDx + dx;
		totalDy = totalDy + dy;
	}

	this->averageFlowDistanceX = totalDx / groundPlaneFlowTails.size();
	this->averageFlowDistanceY = totalDy / groundPlaneFlowTails.size();
	this->angleOfMotion = atan(this->averageFlowDistanceX / this->averageFlowDistanceY) * 180 / 3.14159265;
}


class Cuboid
{
public:
	Blob blob;
	double cuboidLength, cuboidWidth, cuboidHeight, angleOfMotion;
	Point3d bottomLeftCorner, centroid, b1, b2, b3, b4, t1, t2, t3, t4;
	Point2d centroid_imgpt;


	void setOptimizedCuboidParams(double length, double width, double height, double angleOfMotion);
	void Cuboid::moveCuboid(double distanceX, double distanceY, double distanceZ = 0.0);

	Cuboid(Blob blob, double initialCuboidLength, double initialCuboidWidth, double initialCuboidHeight);

	~Cuboid();

};
Cuboid::~Cuboid() {}

Cuboid::Cuboid(Blob blob, double initialCuboidLength, double initialCuboidWidth, double initialCuboidHeight)
{
	this->blob = blob;
	this->cuboidLength = initialCuboidLength;
	this->cuboidWidth = initialCuboidWidth;
	this->cuboidHeight = initialCuboidHeight;
	this->angleOfMotion = this->blob.angleOfMotion;

	Point3f point = findWorldPoint(this->blob.bottomLeftCorner, cameraMatrix, rotationMatrix, translationVector);
	this->bottomLeftCorner = point;

	this->b1 = Point3d(this->bottomLeftCorner.x, this->bottomLeftCorner.y, 0.0);
	this->b2 = Point3d(this->bottomLeftCorner.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y - (this->cuboidWidth* sin(this->angleOfMotion)), 0.0);
	this->b3 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)), 0.0);
	this->b4 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)) +(this->cuboidWidth * sin(this->angleOfMotion)), 0.0);

	this->t1 = Point3d(this->bottomLeftCorner.x, this->bottomLeftCorner.y, this->cuboidHeight);
	this->t2 = Point3d(this->bottomLeftCorner.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y - (this->cuboidWidth* sin(this->angleOfMotion)), this->cuboidHeight);
	this->t3 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)), this->cuboidHeight);
	this->t4 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), this->cuboidHeight);

	this->centroid = Point3d((this->b1.x + this->b2.x) / 2, (this->b1.y + this->b3.y) / 2, this->cuboidHeight / 2);

	vector<Point3f> object_points;
	vector<Point2f> image_points;
	object_points.push_back(this->centroid);

	projectPoints(object_points, rotationVector, translationVector, cameraMatrix, distCoeffs, image_points);
	this->centroid_imgpt = image_points[0];

}

void Cuboid::setOptimizedCuboidParams(double length, double width, double height, double angleOfMotion)
{
	this->cuboidLength = length;
	this->cuboidWidth = width;
	this->cuboidHeight = height;
	this->angleOfMotion = angleOfMotion;

}

void Cuboid::moveCuboid(double distanceX, double distanceY, double distanceZ)
{

	this->bottomLeftCorner.x = this->bottomLeftCorner.x + distanceX;
	this->bottomLeftCorner.y = this->bottomLeftCorner.y + distanceY;
	this->bottomLeftCorner.z = this->bottomLeftCorner.z + distanceZ;

	this->b1 = Point3d(this->bottomLeftCorner.x, this->bottomLeftCorner.y, 0.0);
	this->b2 = Point3d(this->bottomLeftCorner.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y - (this->cuboidWidth* sin(this->angleOfMotion)), 0.0);
	this->b3 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)), 0.0);
	this->b4 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), 0.0);

	this->t1 = Point3d(this->bottomLeftCorner.x, this->bottomLeftCorner.y, this->cuboidHeight);
	this->t2 = Point3d(this->bottomLeftCorner.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y - (this->cuboidWidth* sin(this->angleOfMotion)), this->cuboidHeight);
	this->t3 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)), this->cuboidHeight);
	this->t4 = Point3d(this->bottomLeftCorner.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->bottomLeftCorner.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), this->cuboidHeight);

	this->centroid = Point3d((this->b1.x + this->b2.x) / 2, (this->b1.y + this->b3.y) / 2, this->cuboidHeight / 2);

	vector<Point3f> object_points;
	vector<Point2f> image_points;
	object_points.push_back(this->centroid);

	projectPoints(object_points, rotationVector, translationVector, cameraMatrix, distCoeffs, image_points);
	this->centroid_imgpt = image_points[0];

}
class Track
{
public:

	Track();
	~Track();
	Scalar trackColor = Scalar(rand() % 255, rand() % 255, rand() % 255);
	double averageFlowX, averageFlowY;
	bool trackUpdated, beingTracked = true;
	int matchCount = 0, noMatchCounter, trackNumber;
	vector<Blob> blobs;
	vector<Cuboid> cuboids;

	void addBlobToTrack(Blob);
	void addCuboidToTrack(Cuboid cuboid);
	void drawRects(Mat);
	void drawTrackFeatures(Mat);
	void drawTrack(Mat);
	void drawCuboid(Mat);
	void drawCuboidInfo(Mat);


};

Track::Track()
{
	this->trackNumber = 0;
	this->noMatchCounter = 0;
	this->beingTracked = true;
	this->trackUpdated = false;
};
Track::~Track() {};


void Track::addBlobToTrack(Blob blob)
{

	this->blobs.push_back(blob);
}

void Track::addCuboidToTrack(Cuboid cuboid)
{
	this->cuboids.push_back(cuboid);	

}
void Track::drawRects(Mat outputFrame)
{
	rectangle(outputFrame, this->blobs.back().Bounding_Rectangle,Scalar(this->trackColor) , 1, CV_AA);
	circle(outputFrame, this->blobs.back().bottomLeftCorner, 2, Scalar(0, 0, 255), -1, CV_AA);
}

void Track::drawTrackFeatures(Mat outputFrame)
{
	for (int i = 0; i < this->blobs.back().featurePoints.size(); i++)
	{
		//arrowedLine(frame, this->trackBlobs.back().featurePoints[i], this->trackBlobs.back().flowPoints[i], this->trackColor, 1, CV_AA);
		circle(outputFrame, this->blobs.back().featurePoints[i], 1, this->trackColor, -1, CV_AA);


	}
}

void Track::drawTrack(Mat outputFrame)
{
	
	circle(outputFrame, this->blobs.back().center, 1, Scalar(0, 0, 255), -1, CV_AA);

	for (int i = 0; i < min((int)this->blobs.size(), 10); i++)
	{
		line(outputFrame, this->blobs.rbegin()[i].center, this->blobs.rbegin()[i + 1].center, this->trackColor, 2, CV_AA);
	}

}

void Track::drawCuboid(Mat outputFrame)
{
	vector<Point3d> objectPoints;
	objectPoints.push_back(this->cuboids.back().centroid);
	objectPoints.push_back(this->cuboids.back().b1);
	objectPoints.push_back(this->cuboids.back().b2);
	objectPoints.push_back(this->cuboids.back().b3);
	objectPoints.push_back(this->cuboids.back().b4);
	objectPoints.push_back(this->cuboids.back().t1);
	objectPoints.push_back(this->cuboids.back().t2);
	objectPoints.push_back(this->cuboids.back().t3);
	objectPoints.push_back(this->cuboids.back().t4);
	objectPoints.push_back(this->cuboids.back().centroid);

	vector<Point2d> imagePoints;
	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	circle(outputFrame, imagePoints[0], 2.5, this->trackColor, -1, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[2], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[4], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[4], imagePoints[3], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[1], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[5], imagePoints[6], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[6], imagePoints[8], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[8], imagePoints[7], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[7], imagePoints[5], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[5], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[6], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[7], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[4], imagePoints[8], this->trackColor, 1, CV_AA);

	

}

void Track::drawCuboidInfo(Mat outputFrame)
{
	vector<Point3d> objectPoints;
	objectPoints.push_back(this->cuboids.back().centroid);
	objectPoints.push_back(Point3d((this->cuboids.back().b1.x + (this->cuboids.back().cuboidWidth * this->cuboids.back().angleOfMotion) / 3), (this->cuboids.back().b1.y), (this->cuboids.back().b1.z + this->cuboids.back().cuboidHeight * 2 / 3)));
	objectPoints.push_back(Point3d((this->cuboids.back().b1.x + (this->cuboids.back().cuboidWidth * this->cuboids.back().angleOfMotion) * 2/3), (this->cuboids.back().b1.y), (this->cuboids.back().b1.z + this->cuboids.back().cuboidHeight/ 3)));

	vector<Point2d> imagePoints;
	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	circle(outputFrame, imagePoints[0], 2.5, this->trackColor, -1, CV_AA);

	if (this->matchCount > 5)
	{
		rectangle(outputFrame, imagePoints[1], imagePoints[2], this->trackColor, -1, CV_AA);
		putText(outputFrame, to_string(this->trackNumber), imagePoints[1], CV_FONT_HERSHEY_SIMPLEX, this->cuboids.back().cuboidWidth / 5, Scalar(0.0, 0.0, 0.0), 1, CV_AA);

	}
	putText(outputFrame, "X: " + to_string(this->cuboids.back().centroid.x), Point2d(imagePoints[0].x + 5, imagePoints[0].y), CV_FONT_HERSHEY_SIMPLEX, 0.3, this->trackColor, 1, CV_AA);

	putText(outputFrame, "Y: " + to_string(this->cuboids.back().centroid.y), Point2d(imagePoints[0].x + 5, imagePoints[0].y + 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, this->trackColor, 1, CV_AA);

}


vector<Track> tracks;
int trackCount = 0;
double initialCuboidLength = 4.620, initialCuboidWidth = 1.775, initialCuboidHeight = 1.475;
void matchBlobs(vector<Blob> &frameBlobs, Mat currentFrame, Mat nextFrame);

vector<Point2f> mouseCallBackPoints;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		cout << "Clicked at: " << Point2f(x, y) << endl;
		mouseCallBackPoints.push_back(Point2f(x, y));
	}
}

int main(void)
{
	cout << "Vehicle Speed Estimation Using Optical Flow And 3D Modeling" << endl; cout << endl;
	cout << "Github: https://github.com/IndySupertramp/Vehicle-Speed-Estimation-Using-Optical-Flow-And-3D-Modeling" << endl; cout << endl;

	FileStorage fs("parameters.yml", FileStorage::READ);
	fs["Camera Matrix"] >> cameraMatrix;
	fs["Distortion Coefficients"] >> distCoeffs;
	fs["Rotation Vector"] >> rotationVector;
	fs["Translation Vector"] >> translationVector;;
	fs["Rotation Matrix"] >> rotationMatrix;

	cout << "Camera Matrix: " << endl << cameraMatrix << endl; cout << endl;
	cout << "Distortion Coefficients: " << endl << distCoeffs << endl; cout << endl;
	cout << "Rotation Vector" << endl << rotationVector << endl; cout << endl;
	cout << "Translation Vector: " << endl << translationVector << endl; cout << endl;
	cout << "Rotation Matrix: " << endl << rotationMatrix << endl; cout << endl;

	Mat frame1, frame2, frame1_copy, frame2_copy, frame1_gray, frame2_gray, frame1_blur, frame2_blur, morph, frame1_copy2, frame2_copy2, frame1_copy3, frame2_copy3, diff, thresh;
	VideoCapture capture;
	capture.open("traffic_chiangrak2.MOV");

	if (!capture.isOpened())
	{
		cout << "Error loading file." << endl;
		_getch;
		return -1;
	}

	double frame_rate = capture.get(CV_CAP_PROP_FPS);
	double total_frame_count = capture.get(CV_CAP_PROP_FRAME_COUNT);
	double frame_height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	double frame_width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	Size frameSize(frame_width, frame_height);

	double current_position;

	cout << "Video frame rate: " << frame_rate << endl;
	cout << "Video total frame count: " << total_frame_count << endl;
	cout << "Video frame height: " << frame_height << endl;
	cout << "Video frame width: " << frame_width << endl;

	Size frame_size(frame_width, frame_height);


	capture.read(frame1);
	capture.read(frame2);

	int frame_count = 1;
	bool first_frame = true;
	int key = 0;

	while (capture.isOpened() && key != 27)
	{
		frame1_copy = frame1.clone();
		frame2_copy = frame2.clone();

		//imshow("Original", frame1_copy);

		cvtColor(frame1_copy, frame1_gray, CV_BGR2GRAY);
		cvtColor(frame2_copy, frame2_gray, CV_BGR2GRAY);

		//imshow("Grayscale", frame1_gray);


		GaussianBlur(frame1_gray, frame1_blur, Size(5, 5), 0);
		GaussianBlur(frame2_gray, frame2_blur, Size(5, 5), 0);

		//imshow("Blur", frame1_blur);

		absdiff(frame1_blur, frame2_blur, diff);

		//imshow("Difference Image", diff);

		threshold(diff, thresh, 50, 255.0, CV_THRESH_BINARY);
		
		//imshow("Threshold", thresh);
		morph = thresh.clone();

		for (int i = 0; i < 6; i++)
		{
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			erode(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
		}
		//imshow("Morphed", morph);

		vector<vector<Point> > contours;
		findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			
		Mat img_contour(frameSize, CV_8UC3, Scalar(0, 0, 0));
		drawContours(img_contour, contours, -1, Scalar(255, 255, 255), 1, CV_AA);
		//imshow("Contours", img_contour);

		vector<vector<Point> > convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{

			convexHull(contours[i], convexHulls[i]);
		}

		Mat img_convexHulls(frameSize, CV_8UC3, Scalar(0, 0, 0));
		drawContours(img_convexHulls, convexHulls, -1, Scalar(255, 255, 255), -1, CV_AA);

		//imshow("Convex Hulls", img_convexHulls);

		vector<Blob> frameBlobs;
		for (int i = 0; i < convexHulls.size(); i++)
		{
			Blob blob(convexHulls[i]);

			if (blob.area > 300 && (blob.aspectRatio > 0.2 < 4.0) &&
				blob.width > 20 && blob.height > 20 &&
				blob.diagonalSize > 50.0 && (contourArea(convexHulls[i]) / blob.area) > 0.50)
			{
				frameBlobs.push_back(blob);
			}
		}

		Mat imgFrameBlobs(frameSize, CV_8UC3, Scalar(0, 0, 0));
		for (int i = 0; i < frameBlobs.size(); i++)
		{
			vector<vector<Point> > contours;
			contours.push_back(frameBlobs[i].contour);
			drawContours(imgFrameBlobs, contours, -1, Scalar(255, 255, 255), -1);
		}

		//imshow("Frame Blobs", imgFrameBlobs);



		if (first_frame)
		{
			for (int i = 0; i < frameBlobs.size(); i++)
			{
				frameBlobs[i].findFeatures(frame1_gray);
				Track track;
				track.addBlobToTrack(frameBlobs[i]);
				Cuboid cuboid(frameBlobs[i], initialCuboidLength, initialCuboidWidth, initialCuboidHeight);
				track.addCuboidToTrack(cuboid);
				tracks.push_back(track);
			}
		}

		else
		{
			matchBlobs(frameBlobs, frame1_gray, frame2_gray);
		}

		frame2_copy3 = frame2.clone();

		Mat cuboidSimulation(frame_height, frame_width, CV_8UC3, Scalar(230, 230, 230));


		for (int i = 0; i < tracks.size(); i++)
		{
			if (tracks[i].noMatchCounter < 1 && tracks[i].matchCount > 10)
			{
				//tracks[i].drawRects(frame2_copy3);
				tracks[i].drawTrack(frame2_copy3);
				tracks[i].drawTrackFeatures(frame2_copy3);
				tracks[i].drawCuboid(frame2_copy3);
				tracks[i].drawCuboid(cuboidSimulation);
				tracks[i].drawCuboidInfo(frame2_copy3);
				tracks[i].drawCuboidInfo(cuboidSimulation);
			}
			if (tracks[i].trackUpdated == false) tracks[i].noMatchCounter++;
			if (tracks[i].noMatchCounter >= 10) tracks[i].beingTracked = false;
			if (tracks[i].beingTracked == false) tracks.erase(tracks.begin() + i);
		}

		vector<Point3d> worldPoints;
		worldPoints.push_back(Point3d(0.0, 0.0, 0.0));
		worldPoints.push_back(Point3d(2.0, 0.0, 0.0));
		worldPoints.push_back(Point3d(0.0, 2.0, 0.0));
		worldPoints.push_back(Point3d(0.0, 0.0, 2.0));

		vector<Point2d> imagePoints;

		projectPoints(worldPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

		for (int i = 0; i < imagePoints.size(); i++)
		{
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[1], Scalar(255.0, 0.0, 0.0), 1, CV_AA);
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[2], Scalar(0.0, 255.0, 0.0), 1, CV_AA);
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[3], Scalar(0.0, 0.0, 255.0), 1, CV_AA);

			arrowedLine(frame2_copy3, imagePoints[0], imagePoints[1], Scalar(255.0, 0.0, 0.0), 1, CV_AA);
			arrowedLine(frame2_copy3, imagePoints[0], imagePoints[2], Scalar(0.0, 255.0, 0.0), 1, CV_AA);
			arrowedLine(frame2_copy3, imagePoints[0], imagePoints[3], Scalar(0.0, 0.0, 255.0), 1, CV_AA);

		}

		current_position = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;

		rectangle(frame2_copy3, Point(8, 20), Point(120, 35), Scalar(0, 0, 0), -1, CV_AA);
		putText(frame2_copy3, "Frame Count: " + to_string(frame_count), Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		rectangle(frame2_copy3, Point(8, 40), Point(120, 55), Scalar(0, 0, 0), -1, CV_AA);
		putText(frame2_copy3, "Vehicle Count: " + to_string(trackCount), Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		rectangle(frame2_copy3, Point(8, 60), Point(120, 75), Scalar(0, 0, 0), -1, CV_AA);
		putText(frame2_copy3, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		rectangle(cuboidSimulation, Point(8, 20), Point(120, 35), Scalar(0, 0, 0), -1, CV_AA);
		putText(cuboidSimulation, "Frame Count: " + to_string(frame_count), Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		rectangle(cuboidSimulation, Point(8, 40), Point(120, 55), Scalar(0, 0, 0), -1, CV_AA);
		putText(cuboidSimulation, "Vehicle Count: " + to_string(trackCount), Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		rectangle(cuboidSimulation, Point(8, 60), Point(120, 75), Scalar(0, 0, 0), -1, CV_AA);
		putText(cuboidSimulation, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);

		line(frame2_copy3, Point(314, 0), Point(178, 424), Scalar(0, 0, 255), 2, CV_AA);
		line(frame2_copy3, Point(357, 0), Point(544, 424), Scalar(0, 0, 255), 2, CV_AA);
		line(frame2_copy3, Point(338, 0), Point(360, 424), Scalar(255, 0, 0), 1, CV_AA);

		line(cuboidSimulation, Point(314, 0), Point(178, 424), Scalar(0, 0, 255), 2, CV_AA);
		line(cuboidSimulation, Point(357, 0), Point(544, 424), Scalar(0, 0, 255), 2, CV_AA);
		line(cuboidSimulation, Point(338, 0), Point(360, 424), Scalar(255, 0, 0), 1, CV_AA);

		/*rectangle(frame2_copy3, Point((frame2_copy3.cols * 1 / 4) -3, 0), Point(frame2_copy3.cols * 4 / 4, 14), Scalar(0, 0, 0), -1, CV_AA);*/
	
		/*putText(frame2_copy3, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(frame2_copy3.cols * 1 / 4, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);*/

		putText(frame2_copy3, "https://github.com/IndySupertramp/Vehicle-Speed-Estimation-Using-Optical-Flow-And-3D-Modeling", Point(5, frame2_copy3.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar(255, 255, 255), 0.35, CV_AA);

		/*rectangle(cuboidSimulation, Point((frame2_copy3.cols * 1 / 4)-3, 0), Point(frame2_copy3.cols * 4 / 4, 14), Scalar(0, 0, 0), -1, CV_AA);*/

		/*putText(cuboidSimulation, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(frame2_copy3.cols * 1 / 4, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(255, 255, 255), 0.35, CV_AA);*/

		putText(cuboidSimulation, "https://github.com/IndySupertramp/Vehicle-Speed-Estimation-Using-Optical-Flow-And-3D-Modeling", Point(5, frame2_copy3.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, Scalar(0, 0, 0), 0.35, CV_AA);

		namedWindow("Final");
		moveWindow("Final", 0, 0);
		setMouseCallback("Final", CallBackFunc);
		imshow("Final", frame2_copy3);

		namedWindow("Cuboid Simulation");
		moveWindow("Cuboid Simulation", 640, 0);
		imshow("Cuboid Simulation", cuboidSimulation);

		frameBlobs.clear();

		frame1 = frame2.clone();
		capture.read(frame2);

		frame_count++;
		key = waitKey(1000/frame_rate);

		switch(key)
		{
		case  32:	imwrite("Original.jpg", frame1);
					imwrite("Grayscale.jpg", frame1_gray);
					imwrite("Difference.jpg", diff);
					imwrite("Blurred.jpg", frame1_blur);
					imwrite("Threhold.jpg", thresh);
					imwrite("Morphed.jpg", morph);
					imwrite("Contours.jpg", img_contour);
					imwrite("ConvexHulls.jpg", img_convexHulls);
					imwrite("Blobs.jpg", imgFrameBlobs);
					imwrite("CuboidSimulation.jpg", cuboidSimulation);
					imwrite("Final.jpg", frame2_copy3); continue;
		case 27:	return 0;
		}
		first_frame = false;
	}
	_getch();
	return 0;
}

//   ~~ METHOD FOR MATCHING BLOBS~~   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


void matchBlobs(vector<Blob> &frameBlobs, Mat currentFrame, Mat nextFrame)
{

	for (int i = 0; i < tracks.size(); i++)
	{
		tracks[i].blobs.back().findFlow(currentFrame, nextFrame);
		tracks[i].trackUpdated = false;
	}

	for (int i = 0; i < frameBlobs.size(); i++)
	{
		double leastDistance = 100000;
		int index_of_least_distance;
		Point center = frameBlobs[i].center;

		for (int j = 0; j < tracks.size(); j++)
		{
			bool match = true;
			double sumDistances = 0;
			for (int k = 0; k < tracks[j].blobs.back().flowPoints.size(); k++)
			{

				double distance = distanceBetweenPoints(tracks[j].blobs.back().flowPoints[k], center);
				sumDistances = sumDistances + distance;
			}
			double averageDistance = sumDistances / (double)tracks[j].blobs.back().flowPoints.size();
			if (averageDistance < leastDistance)
			{
				leastDistance = averageDistance;
				index_of_least_distance = j;
			}
		}
		if (leastDistance < frameBlobs[i].diagonalSize * 0.5)
		{
			tracks[index_of_least_distance].cuboids.back().moveCuboid(tracks[index_of_least_distance].blobs.back().averageFlowDistanceX, tracks[index_of_least_distance].blobs.back().averageFlowDistanceY);

			frameBlobs[i].findFeatures(currentFrame);
			tracks[index_of_least_distance].addBlobToTrack(frameBlobs[i]);
			tracks[index_of_least_distance].trackUpdated = true;
			tracks[index_of_least_distance].matchCount++;
			tracks[index_of_least_distance].noMatchCounter = 0;
			if (tracks[index_of_least_distance].matchCount == 10)
			{
				trackCount++;
				tracks[index_of_least_distance].trackNumber = trackCount;
			}
		}
		else
		{
			frameBlobs[i].findFeatures(currentFrame);
			Track track;
			track.addBlobToTrack(frameBlobs[i]);
			Cuboid cuboid(frameBlobs[i], initialCuboidLength, initialCuboidWidth, initialCuboidHeight);
			track.addCuboidToTrack(cuboid);
			tracks.push_back(track);
		}
	}
}
////////////////////////////// ~~ METHOD FOR CALCULATING 3D POINT FROM 2D POINT ~~  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Point3f findWorldPoint(Point2f imagePoint, Mat cameraMatrix, Mat rotationMatrix, Mat translationVector)
{
	Mat imagePointHV = Mat::ones(3, 1, cv::DataType<double>::type);
	imagePointHV.at<double>(0, 0) = imagePoint.x;
	imagePointHV.at<double>(1, 0) = imagePoint.y;
	Mat tempMat, tempMat2;
	double s;
	tempMat = rotationMatrix.inv() * cameraMatrix.inv() * imagePointHV;
	tempMat2 = rotationMatrix.inv() * translationVector;
	s = tempMat2.at<double>(2, 0);
	s /= tempMat.at<double>(2, 0);
	Mat worldPointHV = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePointHV - translationVector);

	Point3f worldPoint;
	worldPoint.x = worldPointHV.at<double>(0, 0);
	worldPoint.y = worldPointHV.at<double>(1, 0);
	worldPoint.z = 0.0;

	return worldPoint;
}

 //   ~~ METHOD FOR FINDING DISTANCE BETWEEN TWO IMAGE POINTS ~~   ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
double distanceBetweenPoints(Point2f point1, Point point2)
{
	point1 = (Point)point1;
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}


double static distanceBetweenPoints(Point2f point1, Point2f point2)
{
	point1 = (Point)point1;
	point2 = (Point)point2;
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}
