#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <string>

using namespace cv;
using namespace std;

Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

double distanceBetweenPoints(Point2f point1, Point point2);
double distanceBetweenPoints(Point2f point1, Point2f point2);
Point3d findGroundPlanePoint(Point2d point, Mat cameraMatrix, Mat rotationVector, Mat translationVector);

class Blob
{
public:
	vector<Point> contour;
	Rect Bounding_Rectangle;
	RotatedRect Rotated_Rect;
	double area, width, height, diagonalSize, aspectRatio, averageFlowDistanceX, averageFlowDistanceY, angleOfMotion;
	Point topRightCorner, bottomRightCorner;
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
		Point3d point = findGroundPlanePoint(Point2d(featurePoints[i]), cameraMatrix, rotationVector, translationVector);
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
		Point3d point = findGroundPlanePoint(Point2d(flowPoints[i]), cameraMatrix, rotationVector, translationVector);
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
	double cuboidLength, cuboidWidth, cuboidHeight, angleOfMotion, averageFlowX, averageFlowY;
	Point3d centroid, b1, b2, b3, b4, t1, t2, t3, t4;


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

	Point3d point = findGroundPlanePoint(this->blob.center, cameraMatrix, rotationVector, translationVector);
	this->centroid.x = point.x;
	this->centroid.y = point.y;
	this->centroid.z = point.z + (cuboidHeight / 2);
	double dx = ((this->cuboidWidth / 2) * cos(this->blob.angleOfMotion)) - ((this->cuboidLength / 2) * sin(this->blob.angleOfMotion));

	double dy = ((this->cuboidWidth / 2) * sin(this->blob.angleOfMotion)) + ((this->cuboidLength / 2) * cos(this->blob.angleOfMotion));

	double dz = cuboidHeight / 2;

	this->b1 = Point3d(this->centroid.x + dx, this->centroid.y + dy, this->centroid.z - dz);
	this->b2 = Point3d(this->centroid.x + dx, this->centroid.y - dy, this->centroid.z - dz);
	this->b3 = Point3d(this->centroid.x - dx, this->centroid.y - dy, this->centroid.z - dz);
	this->b4 = Point3d(this->centroid.x - dx, this->centroid.y + dy, this->centroid.z - dz);

	this->t1 = Point3d(this->centroid.x + dx, this->centroid.y + dy, this->centroid.z + dz);
	this->t2 = Point3d(this->centroid.x + dx, this->centroid.y - dy, this->centroid.z + dz);
	this->t3 = Point3d(this->centroid.x - dx, this->centroid.y - dy, this->centroid.z + dz);
	this->t4 = Point3d(this->centroid.x - dx, this->centroid.y + dy, this->centroid.z + dz);

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

	this->centroid.x = this->centroid.x + distanceX;
	this->centroid.y = this->centroid.y + distanceY;
	this->centroid.z = this->centroid.z + distanceZ;

	double dx = ((this->cuboidWidth / 2) * cos(this->blob.angleOfMotion)) - ((this->cuboidLength / 2) * sin(this->blob.angleOfMotion));

	double dy = ((this->cuboidWidth / 2) * sin(this->blob.angleOfMotion)) + ((this->cuboidLength / 2) * cos(this->blob.angleOfMotion));

	double dz = cuboidHeight / 2;

	this->b1 = Point3d(this->centroid.x + dx, this->centroid.y + dy, this->centroid.z - dz);
	this->b2 = Point3d(this->centroid.x + dx, this->centroid.y - dy, this->centroid.z - dz);
	this->b3 = Point3d(this->centroid.x - dx, this->centroid.y - dy, this->centroid.z - dz);
	this->b4 = Point3d(this->centroid.x - dx, this->centroid.y + dy, this->centroid.z - dz);

	this->t1 = Point3d(this->centroid.x + dx, this->centroid.y + dy, this->centroid.z + dz);
	this->t2 = Point3d(this->centroid.x + dx, this->centroid.y - dy, this->centroid.z + dz);
	this->t3 = Point3d(this->centroid.x - dx, this->centroid.y - dy, this->centroid.z + dz);
	this->t4 = Point3d(this->centroid.x - dx, this->centroid.y + dy, this->centroid.z + dz);


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
	vector<Point> center;


	void addBlobToTrack(Blob);
	void addCuboidToTrack(Cuboid cuboid);
	void drawRects(Mat);
	void drawTrackFeatures(Mat);
	void drawTrack(Mat);
	void drawTrackInfo(Mat);
	void drawCuboid(Mat);




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
	rectangle(outputFrame, this->blobs.back().Bounding_Rectangle, this->trackColor, 1, CV_AA);
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

void Track::drawTrackInfo(Mat outputFrame)
{
	if (this->matchCount > 5)
	{
		Blob blob = this->blobs.back();
		rectangle(outputFrame, blob.topRightCorner, Point(blob.topRightCorner.x + blob.width, blob.topRightCorner.y - blob.diagonalSize / 9), this->trackColor, -1, CV_AA);
		putText(outputFrame, "Track " + to_string(this->trackNumber), blob.topRightCorner, CV_FONT_HERSHEY_SIMPLEX, blob.width / 250, Scalar(255, 255, 255), 1, CV_AA);

	}
}

void Track::drawCuboid(Mat outputFrame)
{
	vector<Point3d> vertices;
	vertices.push_back(this->cuboids.back().b1);
	vertices.push_back(this->cuboids.back().b2);
	vertices.push_back(this->cuboids.back().b3);
	vertices.push_back(this->cuboids.back().b4);
	vertices.push_back(this->cuboids.back().t1);
	vertices.push_back(this->cuboids.back().t2);
	vertices.push_back(this->cuboids.back().t3);
	vertices.push_back(this->cuboids.back().t4);

	vector<Point2d> imagePoints;
	projectPoints(vertices, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	line(outputFrame, imagePoints[0], imagePoints[1], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[2], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[3], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[0], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[4], imagePoints[5], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[5], imagePoints[6], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[6], imagePoints[7], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[7], imagePoints[4], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[0], imagePoints[4], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[5], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[6], this->trackColor, 1, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[7], this->trackColor, 1, CV_AA);


}

vector<Track> tracks;
int trackCount = 0;
double initialCuboidLength = 4.620, initialCuboidWidth = 1.775, initialCuboidHeight = 1.475;
void matchBlobs(vector<Blob> &frameBlobs, Mat currentFrame, Mat nextFrame);
int main(void)
{
	cout << "Vehicle Speed Estimation Using Optical Flow And 3D Modeling" << endl; cout << endl;

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

	Mat frame1, frame2, frame1_copy, frame2_copy, frame1_copy2, frame2_copy2, frame1_copy3, frame2_copy3, diff, thresh; 	VideoCapture capture;
	capture.open("traffic_chiangrak.MOV");

	if (!capture.isOpened())
	{
		cout << "Error loading file." << endl;
		_getch;
		return -1;
	}

	double frame_rate = capture.get(CV_CAP_PROP_FPS);
	double total_frame_count = capture.get(CV_CAP_PROP_FRAME_COUNT);
	double frame_height = 484;
	double frame_width = 640;
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
	capture.set(CV_CAP_PROP_FRAME_WIDTH, frame_width);

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

		cvtColor(frame1_copy, frame1_copy, CV_BGR2GRAY);
		cvtColor(frame2_copy, frame2_copy, CV_BGR2GRAY);

		//imshow("Grayscale", frame1_copy);

		frame1_copy2 = frame1_copy.clone();
		frame2_copy2 = frame2_copy.clone();

		GaussianBlur(frame1_copy, frame1_copy, Size(5, 5), 0);
		GaussianBlur(frame2_copy, frame2_copy, Size(5, 5), 0);

		absdiff(frame1_copy, frame2_copy, diff);

		//imshow("Difference Image", diff);

		threshold(diff, thresh, 45, 255.0, CV_THRESH_BINARY);

		//imshow("Threshold", thresh);
		for (int i = 0; i < 3; i++)
		{
			dilate(thresh, thresh, getStructuringElement(MORPH_RECT, Size(5, 5)));
			dilate(thresh, thresh, getStructuringElement(MORPH_RECT, Size(5, 5)));

			erode(thresh, thresh, getStructuringElement(MORPH_RECT, Size(5, 5)));
		}
		//imshow("Morphed", thresh);

		Mat thresh_copy = thresh.clone();
		vector<vector<Point> > contours;
		findContours(thresh_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		Mat img_contour(thresh_copy.size(), CV_8UC3, Scalar(0, 0, 0));
		drawContours(img_contour, contours, -1, Scalar(255, 255, 255), -1);
		//imshow("Contours", img_contour);

		vector<vector<Point> > contours_poly(contours.size());

		vector<vector<Point> > convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{

			convexHull(contours[i], convexHulls[i]);
		}

		Mat img_convexHulls(thresh_copy.size(), CV_8UC3, Scalar(0, 0, 0));
		drawContours(img_convexHulls, convexHulls, -1, Scalar(255, 255, 255), -1);

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

		Mat imgFrameBlobs(thresh_copy.size(), CV_8UC3, Scalar(0, 0, 0));
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
				frameBlobs[i].findFeatures(frame1_copy2);
				Track track;
				track.addBlobToTrack(frameBlobs[i]);
				Cuboid cuboid(frameBlobs[i], initialCuboidLength, initialCuboidWidth, initialCuboidHeight);
				track.addCuboidToTrack(cuboid);
				tracks.push_back(track);
			}
		}

		else
		{
			matchBlobs(frameBlobs, frame1_copy2, frame2_copy2);
		}

		frame2_copy3 = frame2.clone();

		Mat cuboidSimulation((int)frame_height, (int)frame_width, CV_8UC3, Scalar(0, 0, 0));


		for (int i = 0; i < tracks.size(); i++)
		{
			if (tracks[i].noMatchCounter < 1 && tracks[i].matchCount > 10)
			{
				tracks[i].drawRects(frame2_copy3);
				tracks[i].drawTrack(frame2_copy3);
				tracks[i].drawTrackInfo(frame2_copy3);
				tracks[i].drawTrackFeatures(frame2_copy3);
				tracks[i].drawCuboid(frame2_copy3);
				tracks[i].drawCuboid(cuboidSimulation);
			}
			if (tracks[i].trackUpdated == false) tracks[i].noMatchCounter++;
			if (tracks[i].noMatchCounter >= 10) tracks[i].beingTracked = false;
			if (tracks[i].beingTracked == false) tracks.erase(tracks.begin() + i);
		}

		vector<Point3d> worldPoints;
		worldPoints.push_back(Point3d(0.0, 0.0, 0.0));
		worldPoints.push_back(Point3d(5.0, 0.0, 0.0));
		worldPoints.push_back(Point3d(0.0, 5.0, 0.0));
		worldPoints.push_back(Point3d(0.0, 0.0, -5.0));

		vector<Point2d> imagePoints;

		projectPoints(worldPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

		for (int i = 0; i < imagePoints.size(); i++)
		{
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[1], Scalar(255, 0.0, 0.0), 1, CV_AA);
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[2], Scalar(0.0, 255, 0.0), 1, CV_AA);
			arrowedLine(cuboidSimulation, imagePoints[0], imagePoints[3], Scalar(0.0, 0.0, 255), 1, CV_AA);	
		}

		current_position = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;

		rectangle(frame2_copy3, Point(8, 20), Point(120, 35), Scalar(0, 0, 0), -1, CV_AA);
		rectangle(frame2_copy3, Point(8, 20), Point(120, 35), Scalar(0, 255, 0), 1, CV_AA);
		putText(frame2_copy3, "Frame Count: " + to_string(frame_count), Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 255, 0), 0.35, CV_AA);

		rectangle(frame2_copy3, Point(8, 40), Point(120, 55), Scalar(0, 0, 0), -1, CV_AA);
		rectangle(frame2_copy3, Point(8, 40), Point(120, 55), Scalar(0, 255, 0), 1, CV_AA);
		putText(frame2_copy3, "Vehicle Count: " + to_string(trackCount), Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 255, 0), 0.35, CV_AA);

		rectangle(frame2_copy3, Point(8, 60), Point(120, 75), Scalar(0, 0, 0), -1, CV_AA);
		rectangle(frame2_copy3, Point(8, 60), Point(120, 75), Scalar(0, 255, 0), 1, CV_AA);
		putText(frame2_copy3, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 255, 0), 0.35, CV_AA);

		putText(frame2_copy3, "Vehicle Detection by Indrajeet Datta", Point(frame2_copy3.cols * 2 / 3, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 255), 0.35, CV_AA);

		imshow("Final", frame2_copy3);
		imshow("Cuboid Simulation", cuboidSimulation);

		frameBlobs.clear();

		frame1 = frame2.clone();
		capture.read(frame2);
		key = waitKey(1000 / frame_rate);

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
Point3d findGroundPlanePoint(Point2d point, Mat cameraMatrix, Mat rotationVector, Mat translationVector)
{
	Mat RT;
	hconcat(rotationVector, translationVector, RT);
	Mat projectionMatrix = cameraMatrix * RT;

	double p11 = projectionMatrix.at<double>(0, 0),
		p12 = projectionMatrix.at<double>(0, 1),
		p14 = projectionMatrix.at<double>(0, 3),
		p21 = projectionMatrix.at<double>(1, 0),
		p22 = projectionMatrix.at<double>(1, 1),
		p24 = projectionMatrix.at<double>(1, 3),
		p31 = projectionMatrix.at<double>(2, 0),
		p32 = projectionMatrix.at<double>(2, 1),
		p34 = projectionMatrix.at<double>(2, 3);

	Mat homographyMatrix = (Mat_<double>(3, 3) << p11, p12, p14, p21, p22, p24, p31, p32, p34);

	Mat inverseHomographyMatrix = homographyMatrix.inv();

	Mat matPoint2D = (Mat_<double>(3, 1) << point.x, point.y, 1);
	Mat matPoint3Dw = inverseHomographyMatrix*matPoint2D;
	double w = matPoint3Dw.at<double>(2, 0);
	Mat matPoint3D;
	divide(w, matPoint3Dw, matPoint3D);
	Point3f point3D;
	point3D.x = -1 * matPoint3D.at<double>(0, 0);
	point3D.y = -1 * matPoint3D.at<double>(1, 0);
	point3D.z = 0.0;
	return point3D;
}////

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


