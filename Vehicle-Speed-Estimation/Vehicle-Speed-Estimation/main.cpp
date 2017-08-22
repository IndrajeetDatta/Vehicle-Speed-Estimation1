#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <string>
#include <ctime>

using namespace cv;
using namespace std;

Mat cameraMatrix_, distCoeffs_, rotationVector_, translationVector_, rotationMatrix_, inverseHomographyMatrix_;

double distanceBetweenPoints(Point2f point1, Point point2);
double distanceBetweenPoints(Point2f point1, Point2f point2);
double distancePoint3dZconst(Point3f point1, Point3f point2);
Point3f findWorldPoint(Point2f imagePoint, double zConst, Mat cameraMatrix, Mat rotationMatrix, Mat translationVector);
double frame_rate;
double time_elapsed;
int trackCount = 0;
double initialCuboidLength = 5, initialCuboidWidth = 2, initialCuboidHeight = 1.5;
Scalar WHITE = Scalar(255, 255, 255), BLACK = Scalar(0, 0, 0), BLUE = Scalar(255, 0, 0), GREEN = Scalar(0, 255, 0), RED = Scalar(0, 0, 255), YELLOW = Scalar(0, 255, 255);

class Blob
{
private:
	vector<Point> contour; //Vector of Point of the convex hull of the blob
	Rect Bounding_Rectangle; // Bounding Rectangle of the convex hull of the blob
	double area; // Area of the bounding rectangle
	double width; // Width of the bounding rectangle
	double height; // Height of the bounding rectangle
	double diagonalSize; // Diagonal size of the bounding rectangle
	double aspectRatio; // Aspect Ratio of the bounding rectangle
	double averageFlowDistanceX; // Average of optical flow vector lengths in the X axis
	double averageFlowDistanceY; // Average of optical flow vector lengths is the Y axis
	double angleOfMotion; // Angle of the average optical flow vector
	Point2f center; // Center of the bounding rectangle of the blob with respect to the image coordinates.
	Point2f bottomLeftCorner; // Bottom left corner point of the bounding rectangle
	Point2f topRightCorner; // Top right corner point of the bounding rectangle
	Point2f bottomRightCorner; // Bottom right corner point of the bounding rectangle
	vector<Point2f> featurePoints; // Vector of points to store the feature points
	vector<Point2f> flowPoints; // Vector of points to store the optical flow head points
	vector<Point3f> groundPlaneFlowTails; // Vector of Point3f to store the ground plane coordinates of the tail points of the optical flow vectors
	vector<Point3f> groundPlaneFlowHeads; // Vector of Point3f to store the ground plane coordinates of the head points of the optical flow vectors 
public:
	Blob(vector<Point> contour);
	Blob();
	~Blob();

	vector<Point> getContour() { return this->contour; }
	Rect getBoundingRectangle() { return this->Bounding_Rectangle; }
	double getArea() { return this->area; }
	double getWidth() { return this->width; }
	double getHeight() { return this->height; }
	double getDiagonalSize() { return this->diagonalSize; }
	double getAspectRatio() { return this->aspectRatio; }
	double getAverageFlowDistanceX() { return this->averageFlowDistanceX; }
	double getAverageFlowDistanceY() { return this->averageFlowDistanceY; }
	double getAngleOfMotion() { return this->angleOfMotion; }
	Point2f getCenter() { return this->center; }
	Point2f getBottomLeftCorner() { return this->bottomLeftCorner; }
	Point2f getTopRightCorner() { return this->topRightCorner; }
	Point2f getBottomRightCorner() { return this->bottomRightCorner; }
	vector<Point2f> getFeaturePoints() { return this->featurePoints; }
	vector<Point2f> getFlowPoints() { return this->flowPoints; }
	vector<Point3f> getGroundPlaneFlowTails() { return this->groundPlaneFlowTails; }
	vector<Point3f> getGroundPlaneFlowHeads() { return this->groundPlaneFlowHeads; }



	void findFeatures(Mat);
	void findFlow(Mat, Mat);
};
Blob::Blob(){}
Blob::~Blob(){}

Blob::Blob(vector<Point> contour)
{
	this->contour = contour;
	this->Bounding_Rectangle = boundingRect(this->contour);
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
	Mat mask(currentFrame.size(), CV_8UC1, BLACK);
	drawContours(mask, contours, -1, WHITE, -1);

	goodFeaturesToTrack(currentFrame, featurePoints, 100, 0.01, 5, mask);
	cornerSubPix(currentFrame, featurePoints, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	this->featurePoints = featurePoints;
	for (int i = 0; i < this->featurePoints.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(featurePoints[i]), 0.0, cameraMatrix_, rotationMatrix_, translationVector_);
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
		Point3f point = findWorldPoint(Point2f(flowPoints[i]), 0.0, cameraMatrix_, rotationMatrix_, translationVector_);
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
private:
	Blob blob; // Stores the corresponding blob of the cuboid
	double cuboidLength; // Stores the corresponding blob of the cuboid
	double cuboidWidth; // Stores the width of the cuboid
	double cuboidHeight; // Stores the height of the cuboid
	double angleOfMotion; // Stored the angle of motion of the cuboid from the optical flow vectors of the blob
	Point3f centroid; // Stores the centroid of the cuboidB
	Point3f b1, b2, b3, b4, t1, t2, t3, t4; // Stores the vertices if the cuboid
public:
	Cuboid(Blob blob);
	Cuboid();
	~Cuboid();

	Blob getBlob() { return this->blob; }
	double getCuboidLength() { return this->cuboidLength; }
	double getCuboidWidth() { return this->cuboidWidth; }
	double getCuboidHeight() { return this->cuboidHeight; }
	double getAngleOfMotion() { return this->angleOfMotion; }
	Point3f getCentroid() { return this->centroid; }
	Point3f getB1() { return this->b1; }
	Point3f getB2() { return this->b2; }
	Point3f getB3() { return this->b3; }
	Point3f getB4() { return this->b4; }
	Point3f getT1() { return this->t1; }
	Point3f getT2() { return this->t2; }
	Point3f getT3() { return this->t3; }
	Point3f getT4() { return this->t4; }

	void setBlob(Blob blob);
	void setOptimizedCuboidParams(double length, double width, double height, double angleOfMotion);
	void Cuboid::moveCuboid();
};
Cuboid::Cuboid() {}
Cuboid::~Cuboid() {}
Cuboid::Cuboid(Blob blob)
{
	this->blob = blob;
	this->cuboidLength = initialCuboidLength;
	this->cuboidWidth = initialCuboidWidth;
	this->cuboidHeight = initialCuboidHeight;
	this->angleOfMotion = this->blob.getAngleOfMotion();

	Point3f point = findWorldPoint(this->blob.getBottomLeftCorner(), 0.0, cameraMatrix_, rotationMatrix_, translationVector_);

	this->b1 = Point3f(point.x, point.y, 0.0);

	this->b2 = Point3f(this->b1.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y - (this->cuboidWidth* sin(this->angleOfMotion)), 0.0);

	this->b3 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)), 0.0);

	this->b4 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), 0.0);

	this->t1 = Point3f(this->b1.x, this->b1.y, this->cuboidHeight);

	this->t2 = Point3f(this->b1.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y - (this->cuboidWidth* sin(this->angleOfMotion)), this->cuboidHeight);

	this->t3 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)), this->cuboidHeight);

	this->t4 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), this->cuboidHeight);

	this->centroid = Point3f((this->b1.x + this->b2.x) / 2, (this->b1.y + this->b3.y) / 2, this->cuboidHeight / 2);



}
void Cuboid::setBlob(Blob blob)
{
	this->blob = blob;
}

void Cuboid::setOptimizedCuboidParams(double length, double width, double height, double angleOfMotion)
{
	this->cuboidLength = length;
	this->cuboidWidth = width;
	this->cuboidHeight = height;
	this->angleOfMotion = angleOfMotion;

}

void Cuboid::moveCuboid()
{

	this->b1.x = this->b1.x + this->blob.getAverageFlowDistanceX();
	this->b1.y = this->b1.y + this->blob.getAverageFlowDistanceY();

	this->b2 = Point3f(this->b1.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y - (this->cuboidWidth* sin(this->angleOfMotion)), 0.0);
	this->b3 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)), 0.0);
	this->b4 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), 0.0);

	this->t1 = Point3f(this->b1.x, this->b1.y, this->cuboidHeight);
	this->t2 = Point3f(this->b1.x + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y - (this->cuboidWidth* sin(this->angleOfMotion)), this->cuboidHeight);
	this->t3 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)), this->cuboidHeight);
	this->t4 = Point3f(this->b1.x + (this->cuboidLength * sin(this->angleOfMotion)) + (this->cuboidWidth * cos(this->angleOfMotion)), this->b1.y + (this->cuboidLength * cos(this->angleOfMotion)) + (this->cuboidWidth * sin(this->angleOfMotion)), this->cuboidHeight);

	this->centroid = Point3f((this->b1.x + this->b2.x) / 2, (this->b1.y + this->b3.y) / 2, this->cuboidHeight / 2);

}
class Track
{
private:

public:

	Track(Blob blob);
	~Track();

	vector<Blob> blobs; // A vector of Blob objects that are part of the track
	Cuboid cuboid; // Cuboid object associated with the vector of Blob objects
	vector<Point3f> bottomLeftCorners; // A vector that stores all the ground plane bottom-left corner points of the cuboid in the track
	bool trackUpdated;  // A Boolean variable that stores true if the track is updated
	bool beingTracked = true; // A Boolean variable that stores true if the track is being tracked
	int matchCount = 0; // An integer variable which stores the number of consecutive frames in which a matching blob was found
	int noMatchCount; // An integer variable which stores the number of consecutive frame passed without finding a matching blob.
	int trackNumber; // An integer variable which stores the track number
	Scalar trackColor; // A scalar variable which takes an arbitrary R, G, B value to give each track a unique color while drawing the track or the cuboids on the image

	double measureSpeed(double fps);
	void drawTrack(Mat frame);
	void drawCuboid(Mat);

};

Track::Track(Blob blob)
{
	blobs.push_back(blob);
	Cuboid cuboid(blob);
	bottomLeftCorners.push_back(cuboid.getB1());
	this->cuboid = cuboid;
	this->trackNumber = 0;
	this->noMatchCount = 0;
	this->beingTracked = true;
	this->trackUpdated = false;
	//srand(time(0));
	this->trackColor = Scalar(rand() % 256, rand() % 256, rand() % 256);
};
Track::~Track() {};

double Track::measureSpeed(double fps)
{
	double distance = distancePoint3dZconst(this->bottomLeftCorners.rbegin()[0], this->bottomLeftCorners.rbegin()[1]);
	double time = (1 / fps);
	double speed = (distance / time)*3.6;
	return speed;
}

void Track::drawTrack(Mat outputFrame)
{
	rectangle(outputFrame, this->blobs.back().getBoundingRectangle(), this->trackColor, 2, CV_AA);
	circle(outputFrame, this->blobs.back().getBottomLeftCorner(), 2, RED, -1, CV_AA);
	circle(outputFrame, this->blobs.back().getCenter(), 1, BLUE, -1, CV_AA);


	for (int i = 0; i < min((int)this->blobs.size(), 10); i++)
	{
		line(outputFrame, this->blobs.rbegin()[i].getCenter(), this->blobs.rbegin()[i + 1].getCenter(), this->trackColor, 1, CV_AA);
	}

	if (this->matchCount > 5)
	{
		Blob blob = this->blobs.back();
		rectangle(outputFrame, blob.getTopRightCorner(), Point(blob.getTopRightCorner().x + blob.getWidth(), blob.getTopRightCorner().y - blob.getDiagonalSize() / 9), this->trackColor, -1, CV_AA);

		putText(outputFrame, "Vehicle: " + to_string(this->trackNumber), Point(blob.getTopRightCorner().x + 3, blob.getTopRightCorner().y - 3), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 250, WHITE, 1, CV_AA);

		rectangle(outputFrame, Point(this->blobs.back().getBottomLeftCorner().x + 3, this->blobs.back().getBottomLeftCorner().y + 3), Point(this->blobs.back().getBottomLeftCorner().x + blob.getWidth() - 3, this->blobs.back().getBottomLeftCorner().y + 55), this->trackColor, -1, CV_AA);

		putText(outputFrame, "(" + to_string(this->blobs.back().getBottomLeftCorner().x) + ", " + to_string(this->blobs.back().getBottomLeftCorner().y) + ")", Point2f(this->blobs.back().getBottomLeftCorner().x + 5, this->blobs.back().getBottomLeftCorner().y + 10), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 350, WHITE, 1, CV_AA);

		putText(outputFrame, "A: " + to_string(this->blobs.back().getArea()), Point2f(this->blobs.back().getBottomLeftCorner().x + 5, this->blobs.back().getBottomLeftCorner().y + 20), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 350, WHITE, 1, CV_AA);

		putText(outputFrame, "AR: " + to_string(this->blobs.back().getAspectRatio()), Point2f(this->blobs.back().getBottomLeftCorner().x + 5, this->blobs.back().getBottomLeftCorner().y + 30), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 350, WHITE, 1, CV_AA);

		putText(outputFrame, "W: " + to_string(this->blobs.back().getWidth()), Point2f(this->blobs.back().getBottomLeftCorner().x + 5, this->blobs.back().getBottomLeftCorner().y + 40), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 350, WHITE, 1, CV_AA);

		putText(outputFrame, "H: " + to_string(this->blobs.back().getHeight()), Point2f(this->blobs.back().getBottomLeftCorner().x + 5, this->blobs.back().getBottomLeftCorner().y + 50), CV_FONT_HERSHEY_SIMPLEX, blob.getWidth() / 350, WHITE, 1, CV_AA);

	}

	for (int i = 0; i < this->blobs.back().getFeaturePoints().size(); i++)
	{
		circle(outputFrame, this->blobs.back().getFeaturePoints()[i], 1, GREEN, -1, CV_AA);
	}
}

void Track::drawCuboid(Mat outputFrame)
{
	vector<Point3f> objectPoints;
	objectPoints.push_back(this->cuboid.getCentroid());
	objectPoints.push_back(this->cuboid.getB1());
	objectPoints.push_back(this->cuboid.getB2());
	objectPoints.push_back(this->cuboid.getB3());
	objectPoints.push_back(this->cuboid.getB4());
	objectPoints.push_back(this->cuboid.getT1());
	objectPoints.push_back(this->cuboid.getT2());
	objectPoints.push_back(this->cuboid.getT3());
	objectPoints.push_back(this->cuboid.getT4());
	objectPoints.push_back(Point3f((this->cuboid.getB1().x + this->cuboid.getB2().x) / 2, (this->cuboid.getB1().y + this->cuboid.getB2().y) / 2, this->cuboid.getCuboidHeight() / 2));
	vector<Point2f> imagePoints;
	projectPoints(objectPoints, rotationVector_, translationVector_, cameraMatrix_, distCoeffs_, imagePoints);
	bool inFrame = true;
	for (int i = 0; i < imagePoints.size(); i++)
	{
		if (imagePoints[i].x > 640 || imagePoints[i].y > 424) inFrame = false;
	}
	if (inFrame)
	{
		circle(outputFrame, imagePoints[0], 2.5, BLUE, -1, CV_AA);
		line(outputFrame, imagePoints[1], imagePoints[2], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[2], imagePoints[4], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[4], imagePoints[3], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[3], imagePoints[1], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[5], imagePoints[6], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[6], imagePoints[8], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[8], imagePoints[7], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[7], imagePoints[5], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[1], imagePoints[5], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[2], imagePoints[6], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[3], imagePoints[7], this->trackColor, 2, CV_AA);
		line(outputFrame, imagePoints[4], imagePoints[8], this->trackColor, 2, CV_AA);
		circle(outputFrame, imagePoints[1], 2.5, RED, -1, CV_AA);

		vector<Point3f> objectPoints2;
		vector<Point2f> imagePoints2;
		for (int i = 0; i < this->blobs.rbegin()[1].getGroundPlaneFlowHeads().size(); i++)
		{
			objectPoints2.push_back(this->blobs.rbegin()[1].getGroundPlaneFlowHeads()[i]);
		}
		cout << "Ground plane flow heads size: " << this->blobs.rbegin()[1].getGroundPlaneFlowHeads().size() << endl;
		cout << "Object Points Size: " << objectPoints2.size() << endl;

		projectPoints(objectPoints2, rotationVector_, translationVector_, cameraMatrix_, distCoeffs_, imagePoints2);
		for (int i = 0; i < this->blobs.rbegin()[1].getGroundPlaneFlowHeads().size(); i++)
		{
			circle(outputFrame, imagePoints2[i], 2, RED, -1, CV_AA);
		}

		vector<Point3f> objectPoints3;
		vector<Point2f> imagePoints3;
		for (int i = 0; i < this->blobs.rbegin()[0].getGroundPlaneFlowTails().size(); i++)
		{
			objectPoints3.push_back(this->blobs.rbegin()[0].getGroundPlaneFlowTails()[i]);
		}
		cout << "Ground plane flow tails size: " << this->blobs.rbegin()[0].getGroundPlaneFlowTails().size() << endl;
		cout << "Object Points Size: " << objectPoints2.size() << endl;

		projectPoints(objectPoints3, rotationVector_, translationVector_, cameraMatrix_, distCoeffs_, imagePoints3);
		for (int i = 0; i < this->blobs.rbegin()[0].getGroundPlaneFlowTails().size(); i++)
		{
			circle(outputFrame, imagePoints3[i], 2, GREEN, -1, CV_AA);
		}

		if (this->matchCount > 5)
		{
			circle(outputFrame, imagePoints[9], 10, this->trackColor, -1, CV_AA);
			putText(outputFrame, to_string(this->trackNumber), Point2f(imagePoints[9].x - 7, imagePoints[9].y + 3), CV_FONT_HERSHEY_SIMPLEX, this->cuboid.getCuboidWidth() / 6, WHITE, 1, CV_AA);

		}
		rectangle(outputFrame, Point(imagePoints[1].x + 3, imagePoints[1].y + 5), Point(imagePoints[1].x + 65, imagePoints[1].y + 60), this->trackColor, -1, CV_AA);
		putText(outputFrame, "X: " + to_string(this->cuboid.getB1().x), Point2f(imagePoints[1].x + 5, imagePoints[1].y + 15), CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1, CV_AA);
		putText(outputFrame, "Y: " + to_string(this->cuboid.getB1().y), Point2f(imagePoints[1].x + 5, imagePoints[1].y + 25), CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1, CV_AA);
		putText(outputFrame, "x: " + to_string(imagePoints[1].x), Point2f(imagePoints[1].x + 5, imagePoints[1].y + 35), CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1, CV_AA);
		putText(outputFrame, "y: " + to_string(imagePoints[1].y), Point2f(imagePoints[1].x + 5, imagePoints[1].y + 45), CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1, CV_AA);
		putText(outputFrame, "S: " + to_string(measureSpeed(frame_rate)), Point2f(imagePoints[1].x + 5, imagePoints[1].y + 55), CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 1, CV_AA);
	}

}


vector<Track> tracks;
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

	FileStorage fs("parameters.yml", FileStorage::READ);
	fs["Camera Matrix"] >> cameraMatrix_;
	fs["Distortion Coefficients"] >> distCoeffs_;
	fs["Rotation Vector"] >> rotationVector_;
	fs["Translation Vector"] >> translationVector_;;
	fs["Rotation Matrix"] >> rotationMatrix_;

	cout << "Camera Matrix: " << endl << cameraMatrix_ << endl; cout << endl;
	cout << "Distortion Coefficients: " << endl << distCoeffs_ << endl; cout << endl;
	cout << "Rotation Vector" << endl << rotationVector_ << endl; cout << endl;
	cout << "Translation Vector: " << endl << translationVector_ << endl; cout << endl;
	cout << "Rotation Matrix: " << endl << rotationMatrix_ << endl; cout << endl;

	Mat frame1, frame2, frame1_copy, frame2_copy, frame1_gray, frame2_gray, frame1_blur, frame2_blur, morph, frame1_copy2, frame2_copy2, frame1_copy3, diff, thresh, imgCuboids, imgTracks;
	VideoCapture capture;
	capture.open("traffic_chiangrak2.MOV");

	if (!capture.isOpened())
	{
		cout << "Error loading file." << endl;
		_getch;
		return -1;
	}

	frame_rate = capture.get(CV_CAP_PROP_FPS);
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

		Mat mask(frame_height, frame_width, CV_8UC1, Scalar(1, 1, 1));
		Point mask_points[1][3];
		mask_points[0][0] = Point(0, 0);
		mask_points[0][1] = Point(0, 342);
		mask_points[0][2] = Point(296, 0);
		const Point* ppt[1] = { mask_points[0] };
		int npt[] = { 3 };
		fillPoly(mask, ppt, npt, 1, Scalar(0, 0, 0), CV_AA);
		//imshow("Mask", mask);
		morph = thresh.clone().mul(mask);

		for (int i = 0; i < 6; i++)
		{
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			erode(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
		}
		//imshow("Morphed", morph);

		vector<vector<Point> > contours;
		findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		Mat img_contour(frameSize, CV_8UC3, BLACK);
		drawContours(img_contour, contours, -1, WHITE, 1, CV_AA);
		//imshow("Contours", img_contour);

		vector<vector<Point> > convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{

			convexHull(contours[i], convexHulls[i]);
		}

		Mat img_convexHulls(frameSize, CV_8UC3, BLACK);
		drawContours(img_convexHulls, convexHulls, -1, WHITE, -1, CV_AA);

		//imshow("Convex Hulls", img_convexHulls);

		vector<Blob> frameBlobs;
		for (int i = 0; i < convexHulls.size(); i++)
		{
			Blob blob(convexHulls[i]);

			if (blob.getArea() > 300 && blob.getArea() && (blob.getAspectRatio() > 0.2 < 4.0) &&
				blob.getWidth() > 20 && blob.getHeight() > 20 &&
				blob.getDiagonalSize() > 50.0 && (contourArea(convexHulls[i]) / blob.getArea()) > 0.50)
			{
				frameBlobs.push_back(blob);
			}
		}

		Mat imgFrameBlobs(frameSize, CV_8UC3, BLACK);
		for (int i = 0; i < frameBlobs.size(); i++)
		{
			vector<vector<Point> > contours;
			contours.push_back(frameBlobs[i].getContour());
			drawContours(imgFrameBlobs, contours, -1, WHITE, -1);
		}

		//imshow("Frame Blobs", imgFrameBlobs);



		if (first_frame)
		{
			for (int i = 0; i < frameBlobs.size(); i++)
			{
				frameBlobs[i].findFeatures(frame1_gray);
				Track track(frameBlobs[i]);
				tracks.push_back(track);
			}
		}

		else
		{
			matchBlobs(frameBlobs, frame1_gray, frame2_gray);
		}

		time_elapsed = frame_count / frame_rate;

		imgTracks = frame2.clone();
		imgCuboids = frame2.clone();

		rectangle(imgTracks, Point(8, 20), Point(180, 35), BLACK, -1, CV_AA);
		putText(imgTracks, "Frame Count: " + to_string(frame_count) + " / " + to_string((int)total_frame_count), Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgTracks, Point(8, 40), Point(180, 55), BLACK, -1, CV_AA);
		putText(imgTracks, "Vehicle Count: " + to_string(trackCount), Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgTracks, Point(8, 60), Point(180, 75), BLACK, -1, CV_AA);
		putText(imgTracks, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgTracks, Point(8, 80), Point(180, 95), BLACK, -1, CV_AA);
		putText(imgTracks, "Time Elapsed: " + to_string(time_elapsed), Point(10, 90), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);


		rectangle(imgCuboids, Point(8, 20), Point(180, 35), BLACK, -1, CV_AA);
		putText(imgCuboids, "Frame Count: " + to_string(frame_count) + " / " + to_string((int)total_frame_count), Point(10, 30), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgCuboids, Point(8, 40), Point(180, 55), BLACK, -1, CV_AA);
		putText(imgCuboids, "Vehicle Count: " + to_string(trackCount), Point(10, 50), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgCuboids, Point(8, 60), Point(180, 75), BLACK, -1, CV_AA);
		putText(imgCuboids, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		rectangle(imgCuboids, Point(8, 80), Point(180, 95), BLACK, -1, CV_AA);
		putText(imgCuboids, "Time Elapsed: " + to_string(time_elapsed), Point(10, 90), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);


		line(imgTracks, Point(314, 0), Point(178, 424), YELLOW, 1, CV_AA);
		line(imgTracks, Point(357, 0), Point(544, 424), YELLOW, 1, CV_AA);
		line(imgTracks, Point(338, 0), Point(360, 424), YELLOW, 1, CV_AA);

		line(imgCuboids, Point(314, 0), Point(178, 424), YELLOW, 1, CV_AA);
		line(imgCuboids, Point(357, 0), Point(544, 424), YELLOW, 1, CV_AA);
		line(imgCuboids, Point(338, 0), Point(360, 424), YELLOW, 1, CV_AA);

		line(imgTracks, Point(0, 342), Point(296, 0), RED, 2, CV_AA);
		line(imgCuboids, Point(0, 342), Point(296, 0), RED, 2, CV_AA);


		rectangle(imgTracks, Point((imgTracks.cols * 2 / 3) - 10, 0), Point(imgTracks.cols, 14), BLACK, -1, CV_AA);

		putText(imgTracks, "Vehicle Detection and Tracking", Point((imgCuboids.cols * 2 / 3) - 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		putText(imgTracks, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(5, imgTracks.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 0.35, CV_AA);

		rectangle(imgCuboids, Point((imgCuboids.cols * 2 / 3) - 10, 0), Point(imgCuboids.cols, 14), BLACK, -1, CV_AA);

		putText(imgCuboids, "Cuboid Estimation of Tracked Vehicles", Point((imgCuboids.cols * 2 / 3) - 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		putText(imgCuboids, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(5, imgCuboids.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 0.35, CV_AA);

		for (int i = 0; i < tracks.size(); i++)
		{
			if (tracks[i].noMatchCount < 1 && tracks[i].matchCount > 10)
			{
				tracks[i].drawTrack(imgTracks);
				drawContours(imgTracks, contours, -1, RED, 1, CV_AA);
				drawContours(imgTracks, convexHulls, -1, BLUE, 1, CV_AA);

				tracks[i].drawCuboid(imgCuboids);
			}
			if (tracks[i].trackUpdated == false) tracks[i].noMatchCount++;
			if (tracks[i].noMatchCount >= 10) tracks[i].beingTracked = false;
			if (tracks[i].beingTracked == false) tracks.erase(tracks.begin() + i);
		}

		vector<Point3f> worldPoints;
		worldPoints.push_back(Point3f(0.0, 0.0, 0.0));
		worldPoints.push_back(Point3f(1.0, 0.0, 0.0));
		worldPoints.push_back(Point3f(0.0, 1.0, 0.0));
		worldPoints.push_back(Point3f(0.0, 0.0, 1.0));

		vector<Point2f> imagePoints;

		projectPoints(worldPoints, rotationVector_, translationVector_, cameraMatrix_, distCoeffs_, imagePoints);

		for (int i = 0; i < imagePoints.size(); i++)
		{
			arrowedLine(imgCuboids, imagePoints[0], imagePoints[1], BLUE, 1, CV_AA);
			arrowedLine(imgCuboids, imagePoints[0], imagePoints[2], GREEN, 1, CV_AA);
			arrowedLine(imgCuboids, imagePoints[0], imagePoints[3], RED, 1, CV_AA);

		}

		current_position = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;



		namedWindow("Tracking");
		moveWindow("Tracking", 0, 0);
		imshow("Tracking", imgTracks);

		namedWindow("Cuboids");
		moveWindow("Cuboids", 640, 0);
		setMouseCallback("Cuboids", CallBackFunc);
		imshow("Cuboids", imgCuboids);

		/*VideoWriter outputVideo;
		outputVideo.open("outputVideo.avi", CV_FOURCC('D', 'I', 'V', 'X'), frame_rate, Size(frame_width, frame_height));
		outputVideo.write(imgCuboids);*/

		frameBlobs.clear();

		frame1 = frame2.clone();
		capture.read(frame2);


		key = waitKey(1000 / frame_rate);
		//key = waitKey(0);
		switch (key)
		{
		case  32:	imwrite("frame" + to_string(frame_count) + "Original.jpg", frame1);
			imwrite("frame" + to_string(frame_count) + "Grayscale.jpg", frame1_gray);
			imwrite("frame" + to_string(frame_count) + "Difference.jpg", diff);
			imwrite("frame" + to_string(frame_count) + "Blurred.jpg", frame1_blur);
			imwrite("frame" + to_string(frame_count) + "Threhold.jpg", thresh);
			imwrite("frame" + to_string(frame_count) + "Morphed.jpg", morph);
			imwrite("frame" + to_string(frame_count) + "Contours.jpg", img_contour);
			imwrite("frame" + to_string(frame_count) + "ConvexHulls.jpg", img_convexHulls);
			imwrite("frame" + to_string(frame_count) + "Blobs.jpg", imgFrameBlobs);
			imwrite("frame" + to_string(frame_count) + "Tracking.jpg", imgTracks);
			imwrite("frame" + to_string(frame_count) + "Cuboids.jpg", imgCuboids); continue;
		case 27:	return 0;
		}
		first_frame = false;
		frame_count++;
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
		Point center = frameBlobs[i].getCenter();

		for (int j = 0; j < tracks.size(); j++)
		{
			double sumDistances = 0;
			for (int k = 0; k < tracks[j].blobs.back().getFlowPoints().size(); k++)
			{

				double distance = distanceBetweenPoints(tracks[j].blobs.back().getFlowPoints()[k], center);
				sumDistances = sumDistances + distance;
			}
			double averageDistance = sumDistances / (double)tracks[j].blobs.back().getFlowPoints().size();
			if (averageDistance < leastDistance)
			{
				leastDistance = averageDistance;
				index_of_least_distance = j;
			}
		}
		if (leastDistance < frameBlobs[i].getDiagonalSize() * 0.5)
		{
			tracks[index_of_least_distance].cuboid.setBlob(tracks[index_of_least_distance].blobs.back());
			tracks[index_of_least_distance].cuboid.moveCuboid();
			tracks[index_of_least_distance].bottomLeftCorners.push_back(tracks[index_of_least_distance].cuboid.getCentroid());

			frameBlobs[i].findFeatures(currentFrame);
			tracks[index_of_least_distance].blobs.push_back(frameBlobs[i]);
			tracks[index_of_least_distance].trackUpdated = true;
			tracks[index_of_least_distance].matchCount++;
			tracks[index_of_least_distance].noMatchCount = 0;
			if (tracks[index_of_least_distance].matchCount == 10)
			{
				trackCount++;
				tracks[index_of_least_distance].trackNumber = trackCount;
			}
		}
		else
		{
			frameBlobs[i].findFeatures(currentFrame);
			Track track(frameBlobs[i]);
			tracks.push_back(track);
		}
	}
}
////////////////////////////// ~~ METHOD FOR CALCULATING 3D POINT FROM 2D POINT ~~  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Point3f findWorldPoint(Point2f imagePoint, double zConst, Mat cameraMatrix, Mat rotationMatrix, Mat translationVector)
{
	Mat imagePointHV = Mat::ones(3, 1, cv::DataType<double>::type);
	imagePointHV.at<double>(0, 0) = imagePoint.x;
	imagePointHV.at<double>(1, 0) = imagePoint.y;
	Mat A, B;
	A = rotationMatrix.inv() * cameraMatrix.inv() * imagePointHV;
	B = rotationMatrix.inv() * translationVector;
	double p = A.at<double>(2, 0);
	double q = zConst + B.at<double>(2, 0);
	double s = q / p;
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


double distanceBetweenPoints(Point2f point1, Point2f point2)
{
	point1 = (Point)point1;
	point2 = (Point)point2;
	int intX = abs(point1.x - point2.x);
	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

double static distancePoint3dZconst(Point3f point1, Point3f point2)
{
	double x = abs(point1.x - point2.x);
	double y = abs(point1.y - point2.y);

	return(sqrt(pow(x, 2) + pow(y, 2)));
}