/*  Copyright 2011 AIT Austrian Institute of Technology
*
*   This file is part of OpenTLD.
*
*   OpenTLD is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   OpenTLD is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
*
*/
/*
 * NNClassifier.h
 *
 *  Created on: Nov 16, 2011
 *      Author: Georg Nebehay
 */

#ifndef CFCLASSIFIER_H_
#define CFCLASSIFIER_H_

#include <vector>
#include <set>

#include <opencv/cv.h>

#include "Patch.h"
#include "NormalizedPatch.h"
#include "DetectionResult.h"

using namespace cv;

namespace tld
{

class CFClassifier
{
  enum MODE {
    GRAY   = (1 << 0),
    CN     = (1 << 1),
    CUSTOM = (1 << 2)
  };


  struct CV_EXPORTS Params
  {
    /**
    * \brief Constructor
    */
    Params();

    /**
    * \brief Read parameters from a file
    */
    void read(const FileNode& /*fn*/);

    /**
    * \brief Write parameters to a file
    */
    void write(FileStorage& /*fs*/) const;

    float detect_thresh;         //!<  detection confidence threshold
    float sigma;                 //!<  gaussian kernel bandwidth
    float lambda;                //!<  regularization
    float interp_factor;         //!<  linear interpolation factor for adaptation
    float output_sigma_factor;   //!<  spatial bandwidth (proportional to target)
    float pca_learning_rate;     //!<  compression learning rate
    bool resize;                  //!<  activate the resize feature to improve the processing speed
    bool split_coeff;             //!<  split the training coefficients into two matrices
    bool wrap_kernel;             //!<  wrap around the kernel values
    bool compress_feature;        //!<  activate the pca method to compress the features
    int max_patch_size;           //!<  threshold for the ROI size
    int compressed_size;          //!<  feature size after compression
    int desc_pca;        //!<  compressed descriptors of TrackerKCF::MODE
    int desc_npca;       //!<  non-compressed descriptors of TrackerKCF::MODE
  };
  public:
    bool enabled = true;

    float detectionThreshold = 0.5;

    int *windows;
    float thetaFP;
    float thetaTP;
    DetectionResult *detectionResult;
    std::vector<NormalizedPatch>* falsePositives;
    std::vector<NormalizedPatch>* truePositives;

    CFClassifier();
    virtual ~CFClassifier();

    void release();
    void nextIteration(const cv::Mat &image);
    void nextIteration(const cv::Mat &image, int frameNum);
    float classifyWindow(cv::Rect &r);
    float classifyWindow(const cv::Mat &image, cv::Rect &r);
    std::pair<cv::Rect, float> detect(cv::Rect &r);
    float classifyPatch(NormalizedPatch *patch);
    float classifyBB(const cv::Mat &img, cv::Rect *bb);
    float classifyWindow(const cv::Mat &img, int windowIdx);
    void learn(std::vector<Patch> patches);
    void learn(const Mat &im, std::vector<Patch> patches);
    bool filter(const cv::Mat &img, int windowIdx);
    void read( const cv::FileNode& /*fn*/ );
    void write( cv::FileStorage& /*fs*/ ) const;
    void setFeatureExtractor(void (*f)(const Mat, const Rect, Mat&), bool pca_func = false);
    void learnPositive(const Mat &im, Rect &roi);
    void learnNegative(const Mat &im, Rect &roi);

  protected:

    /*
    * KCF functions and vars
    */
    void createHanningWindow(OutputArray dest, const cv::Size winSize, const int type) const;
    void inline fft2(const Mat src, std::vector<Mat> & dest, std::vector<Mat> & layers_data) const;
    void inline fft2(const Mat src, Mat & dest) const;
    void inline ifft2(const Mat src, Mat & dest) const;
    void inline pixelWiseMult(const std::vector<Mat> src1, const std::vector<Mat>  src2, std::vector<Mat>  & dest, const int flags, const bool conjB=false) const;
    void inline sumChannels(std::vector<Mat> src, Mat & dest) const;
    void inline updateProjectionMatrix(const Mat src, Mat & old_cov,Mat &  proj_matrix,float pca_rate, int compressed_sz,
                                       std::vector<Mat> & layers_pca,std::vector<Scalar> & average, Mat pca_data, Mat new_cov, Mat w, Mat u, Mat v);

    void inline compress(const Mat proj_matrix, const Mat src, Mat & dest, Mat & data, Mat & compressed) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, Mat& patch, MODE desc = GRAY) const;
    bool getSubWindow(const Mat img, const Rect roi, Mat& feat, void (*f)(const Mat, const Rect, Mat& )) const;
    //void extractCN(Mat patch_data, Mat & cnFeatures) const;
    void denseGaussKernel(const float sigma, const Mat , const Mat y_data, Mat & k_data,
                          std::vector<Mat> & layers_data,std::vector<Mat> & xf_data,std::vector<Mat> & yf_data, std::vector<Mat> xyf_v, Mat xy, Mat xyf ) const;
    void calcResponse(const Mat alphaf_data, const Mat kf_data, Mat & response_data, Mat & spec_data) const;
    void calcResponse(const Mat alphaf_data, const Mat alphaf_den_data, const Mat kf_data, Mat & response_data, Mat & spec_data, Mat & spec2_data) const;

    void shiftRows(Mat& mat) const;
    void shiftRows(Mat& mat, int n) const;
    void shiftCols(Mat& mat, int n) const;
#ifdef HAVE_OPENCLa
    bool inline oclTransposeMM(const Mat src, float alpha, UMat &dst);
#endif

    void learnPositive(Rect &roi);
    void learnNegative(Rect &roi);

  private:
    float output_sigma;
    bool learnFirst;
    Rect2d roi;
    Mat hann; 	//hann window filter
    Mat hann_cn; //10 dimensional hann-window filter for CN features,

    Mat y,yf; 	// training response and its FFT
    Mat x; 	// observation and its FFT
    Mat k,kf;	// dense gaussian kernel and its FFT
    Mat kf_lambda; // kf+lambda
    Mat new_alphaf, alphaf;	// training coefficients
    Mat new_alphaf_den, alphaf_den; // for splitted training coefficients
    Mat z; // model
    Mat response; // detection result
    Mat old_cov_mtx, proj_mtx; // for feature compression

    // pre-defined Mat variables for optimization of private functions
    Mat spec, spec2;
    std::vector<Mat> layers;
    std::vector<Mat> vxf,vyf,vxyf;
    Mat xy_data,xyf_data;
    Mat data_temp, compress_data;
    std::vector<Mat> layers_pca_data;
    std::vector<Scalar> average_data;
    Mat img_Patch, image;

    // storage for the extracted features, KRLS model, KRLS compressed model
    Mat X[2],Z[2],Zc[2];

    // storage of the extracted features
    std::vector<Mat> features_pca;
    std::vector<Mat> features_npca;
    std::vector<MODE> descriptors_pca;
    std::vector<MODE> descriptors_npca;

    // optimization variables for updateProjectionMatrix
    Mat data_pca, new_covar,w_data,u_data,vt_data;

    // custom feature extractor
    bool use_custom_extractor_pca;
    bool use_custom_extractor_npca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_pca;
    std::vector<void(*)(const Mat img, const Rect roi, Mat& output)> extractor_npca;

    bool resizeImage; // resize the image whenever needed and the patch size is large

#ifdef HAVE_OPENCLa
    ocl::Kernel transpose_mm_ker; // OCL kernel to compute transpose matrix multiply matrix.
#endif

    int frame;
    float ncc(float *f1, float *f2);


  Params params;

};

} /* namespace tld */
#endif /* NNCLASSIFIER_H_ */
