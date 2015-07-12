/*
	------------------------------------------------------------
	  FACE RECOGNITION USING LINEAR BINARY PATTERNS HISTOGRAMS
	------------------------------------------------------------
	    AUTHORS : Anmol Sharma and Harman Preet Kaur
	AFFILIATION : DAV Institute of Engineering and Technology 
	DESCRIPTION : The code is a part of the project Code4Kids, 
				  an initiative by Google Developers Group in
				  collaboration with Haryana Government, and 
				  Commissionar of Police Ambala and Panchkula. 

				  The module trains an LBPH classifier with
				  the given dataset and determines a decision
				  threshold. The main advantage of using LBPH 
				  is that the the model file can be updated 
				  on the fly without retraining with the whole
				  dataset again, hence dramatically saving 
				  server load. 

				  Snippets of code have been used from existing 
				  open source code which is released in the
				  public domain by their respective owners. 
				  
				  INPUTS:
						A .csv file with paths of images and 
						their respective classes seperated by a
						";". Images are of size 128x128. 
				  OUTPUTS:
				        Trained model. 
		
		LICENSE : GNU GPLv2
      LAST EDIT : 8/7/15 2.27am
*/

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {

	// Open the CSV file. 
	std::ifstream file(filename.c_str(), ifstream::in);

	if (!file) { // Check if file is opened? 
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) { // Store the first line in line
		stringstream liness(line); 
		getline(liness, path, separator); // Read until ";" occurs. This is the path. 

		// Remember, getline modifies both "is" and "str". So "liness" should now only contain an integer 
		// essentially anything after the delimiter ";". Read that into classLabel. 
		getline(liness, classlabel); 

		if (!path.empty() && !classlabel.empty()) {
			
			// Read the image in grayscale.
			Mat image = imread(path, 0);
			if (image.empty())	{
				cout << path << endl; // For debugging purposes. 
				cout << "Could not read image! Please check if its in directory or check path" << endl;
			    break;
		    }

			images.push_back(image);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, const char *argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc != 2) {
		cout << "usage: " << argv[0] << " <csv.ext>" << endl;
		exit(1);
	}

	// Get the path to your CSV.
	string fn_csv = string(argv[1]);

	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;

	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;

	/* JUST FOR DEBUGGING AND TESTING THE MODEL
		WILL NOT BE A PART OF THE FINAL RELEASE VERSION
	*/
	// The following lines simply get the last images from
	// the dataset and remove it from the vector. This is
	// done, so that the training data (which we learn the
	// cv::FaceRecognizer on) and the test data we test
	// the model with, do not overlap.
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();
	labels.pop_back();

	// The following lines create an LBPH model for
	// face recognition and train it with the images and
	// labels read from the given CSV file.
	//
	// The LBPHFaceRecognizer uses Extended Local Binary Patterns
	// (it's probably configurable with other operators at a later
	// point), and has the following default values
	//
	//      radius = 1
	//      neighbors = 8
	//      grid_x = 8
	//      grid_y = 8
	//
	// So if you want a LBPH FaceRecognizer using a radius of
	// 2 and 16 neighbors, call the factory method with:
	//
	//      cv::createLBPHFaceRecognizer(2, 16);
	//
	// And if you want a threshold (e.g. 123.0) call it with its default values:
	//
	//      cv::createLBPHFaceRecognizer(1,8,8,8,123.0)
	//

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	model->train(images, labels);

	// The following line predicts the label of a given
	// test image:
	int predictedLabel = -1;
	double confidence;
	model->predict(testSample, predictedLabel, confidence);

	//
	// To get the confidence of a prediction call the model with:
	//
	//      int predictedLabel = -1;
	//      double confidence = 0.0;
	//      model->predict(testSample, predictedLabel, confidence);
	//
	string result_message = format("Predicted class = %d with 
		                           Confidence = %d/ Actual class = %d.", predictedLabel,
							       confidence, testLabel);
	cout << result_message << endl;
	// Sometimes you'll need to get/set internal model data,
	// which isn't exposed by the public cv::FaceRecognizer.
	// Since each cv::FaceRecognizer is derived from a
	// cv::Algorithm, you can query the data.
	//
	// First we'll use it to set the threshold of the FaceRecognizer
	// to 0.0 without retraining the model. This can be useful if
	// you are evaluating the model:
	//

	//model->set("threshold", 0.0);
	// Now the threshold of this model is set to 0.0. A prediction
	// now returns -1, as it's impossible to have a distance below
	// it
	//predictedLabel = model->predict(testSample);
	//cout << "Predicted class = " << predictedLabel << endl;
	
	// Show some informations about the model, as there's no cool
	// Model data to display as in Eigenfaces/Fisherfaces.
	// Due to efficiency reasons the LBP images are not stored
	// within the model:
	cout << "Model Information:" << endl;
	string model_info = format("\tLBPH(radius=%i, neighbors=%i, grid_x=%i, grid_y=%i, threshold=%g)",
		model->getInt("radius"),
		model->getInt("neighbors"),
		model->getInt("grid_x"),
		model->getInt("grid_y"),
		model->getDouble("threshold"));
	cout << model_info << endl;
	// We could get the histograms for example:
	vector<Mat> histograms = model->getMatVector("histograms");
	// But should I really visualize it? Probably the length is interesting:
	cout << "Size of the histograms: " << histograms[0].total() << endl;
	return 0;
}