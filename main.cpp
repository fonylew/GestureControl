#define WINVER 0x0500
#include <windows.h>
#ifdef ACCESS_MASK
#undef ACCESS_MASK
#endif

#include <iostream>
#include <opencv2/opencv.hpp>
#include <queue>
#include <time.h>
#define M_PI    3.14159265358979323846

using namespace std;
using namespace cv;

struct myFilter {
    Mat mean = (Mat_<float>(3,3) <<
                1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/9.0, 1.0/9.0, 1.0/9.0
               );
    Mat mean2 = (Mat_<float>(5,5) <<
                 1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                 1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                 1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                 1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,
                 1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0,1.0/25.0
                );
    Mat mean3 = (Mat_<float>(4,4) <<
                 1.0/16.0,1.0/16.0,1.0/16.0,1.0/16.0,
                 1.0/16.0,1.0/16.0,1.0/16.0,1.0/16.0,
                 1.0/16.0,1.0/16.0,1.0/16.0,1.0/16.0,
                 1.0/16.0,1.0/16.0,1.0/16.0,1.0/16.0
                );
    Mat prewitt_x = (Mat_<float>(3,3) <<
                     -1, 0, 1,
                     -1, 0, 1,
                     -1, 0, 1
                    );
    Mat prewitt_y = (Mat_<float>(3,3) <<
                     1, 1, 1,
                     0, 0, 0,
                     -1, -1, -1
                    );
    Mat sobel_x = (Mat_<float>(3,3) <<
                   -1, 0, 1,
                   -2, 0, 2,
                   -1, 0, 1
                  );
    Mat sobel_y = (Mat_<float>(3,3) <<
                   1, 2, 1,
                   0, 0, 0,
                   -1, -2, -1
                  );
    Mat roberts_x = (Mat_<float>(2,2) <<
                     0, 1,
                     -1, 0
                    );
    Mat roberts_y = (Mat_<float>(2,2) <<
                     1, 0,
                     0, -1
                    );
    void convolution(Mat& in, Mat& out, Mat& mask) {
        out = Mat(in.rows-(mask.rows-1), in.cols-(mask.cols-1),CV_8UC3);
        for(int i=(mask.rows+1)/2-1; i<in.rows-mask.rows/2; i++) {
            for(int j=(mask.cols+1)/2-1; j<in.cols-mask.cols/2; j++) {
                int k=i-((mask.rows+1)/2-1);
                int l=j-((mask.cols+1)/2-1);
                Vec3f temp = Vec3f(0.0f,0.0f,0.0f);
                for(int a=-(mask.rows-1)/2; a<=mask.rows/2; a++) {
                    for(int b=-(mask.cols-1)/2; b<=mask.cols/2; b++) {
                        int c = a+(mask.rows-1)/2;
                        int d = b+(mask.cols-1)/2;
                        Vec3f temp2 = in.at<Vec3b>(i+a,j+b);
                        temp += temp2*mask.at<float>(c,d);
                    }
                }
                out.at<Vec3b>(k,l) = temp;
            }
        }
    }
    void convolution2(Mat& in, Mat& out, Mat& mask) {
        out = Mat(in.rows, in.cols,CV_8UC3);
        for(int i=0; i<in.rows; i++) {
            for(int j=0; j<in.cols; j++) {
                int k=i;
                int l=j;
                Vec3f temp = Vec3f(0.0f,0.0f,0.0f);
                for(int a=-(mask.rows-1)/2; a<=mask.rows/2; a++) {
                    for(int b=-(mask.cols-1)/2; b<=mask.cols/2; b++) {
                        int c = a+(mask.rows-1)/2;
                        int d = b+(mask.cols-1)/2;
                        if(i+a >= 0 && j+b >= 0 && i+a < in.rows && j+b < in.cols) {
                            Vec3f temp2 = in.at<Vec3b>(i+a,j+b);
                            temp += temp2*mask.at<float>(c,d);
                        }
                    }
                }
                out.at<Vec3b>(k,l) = temp;
            }
        }
    }
    Mat euclidean(Mat& inx, Mat& iny) {
        Mat out = Mat(inx.rows, inx.cols, CV_8UC3);
        for(int i=0; i<out.rows; i++) {
            for(int j=0; j<out.cols; j++) {
                Vec3b temp = Vec3b(0,0,0);
                for(int c=0; c<3; c++) {
                    temp[c] = hypot(inx.at<Vec3b>(i,j)[c], iny.at<Vec3b>(i,j)[c]);
                }
                out.at<Vec3b>(i,j) = temp;
            }
        }
        return out;
    }
    void main() {
        Mat inputImage = imread("north.png", IMREAD_COLOR);
        namedWindow("Original");
        imshow("Original", inputImage);
        Mat outx,outy;
        convolution2(inputImage, outx, prewitt_x);
        convolution2(inputImage, outy, prewitt_y);
        namedWindow("Sobel");
        imshow("Sobel", euclidean(outx, outy));
        convolution2(inputImage, outx, sobel_x);
        convolution2(inputImage, outy, sobel_x);
        namedWindow("Prewitt");
        imshow("Prewitt", euclidean(outx, outy));
        convolution2(inputImage, outx, roberts_x);
        convolution2(inputImage, outy, roberts_y);
        namedWindow("Roberts");
        imshow("Roberts", euclidean(outx, outy));
        Canny(inputImage,outx,100,150);
        namedWindow("Canny");
        imshow("Canny", outx);
    }
} myFilter;

Mat captureFrame, shownCaptureFrame, shownCaptureFrame2, captureFrameOriginal, captureFrameHSV, recognizedImage;
vector<bool*> hashRow, hashColumn;
int pos[] =  {-2,-2};
int cpos[] =  {-2,-2};
vector<Vec3b> maxb;
vector<Vec3b> minb;

void myMouseCallback(int event, int x, int y, int flags, void* userdata) {
    int tmpx,tmpy;

    if(event == EVENT_LBUTTONDOWN) {
        pos[0] = y;
        pos[1] = x;
    }
    if(event == EVENT_LBUTTONUP) {
        if(x<pos[1]){
            tmpx = x;
            x = pos[1];
            pos[1] = tmpx;
        }
        if(y<pos[0]){
            tmpy = y;
            y = pos[0];
            pos[0] = tmpy;
        }

        Vec3b maxc = Vec3b(0,0,0);
        Vec3b minc = Vec3b(255,255,255);
        for(int i=pos[0]/2; i<=y/2; i++) {
            for(int j=pos[1]/2; j<=x/2; j++) {
                for(int c=0; c<3; c++) {
                    maxc[c] = max(maxc[c], captureFrameHSV.at<Vec3b>(i,j)[c]);
                    minc[c] = min(minc[c], captureFrameHSV.at<Vec3b>(i,j)[c]);
                }
            }
        }
        cout << "maxb.push_back(Vec3b(" << (int)maxc[0] << "," << (int)maxc[1] << "," << (int)maxc[2] << "));" << endl;
        cout << "minb.push_back(Vec3b(" << (int)minc[0] << "," << (int)minc[1] << "," << (int)minc[2] << "));" << endl;
        maxb.push_back(maxc);
        minb.push_back(minc);
        recognizedImage = captureFrameHSV(Rect(pos[1]/2,pos[0]/2,(x-pos[1])/2,(y-pos[0])/2)).clone();
        pos[0] = -1;
        pos[1] = -1;
    }
    cpos[0] = y;
    cpos[1] = x;
}

int avgX = -1, avgY = -1;

// 0 - 32767
int smallColor(const Vec3b& color) {
    return (int)(color[0]/8) + 32 * (int)(color[1]/8) + 32 * 32 * (int)(color[2]/8);
}

void recognize() {
    if(pos[0] == -2 || pos[1] == -2) {
        return;
    }
    while(hashRow.size() < captureFrameHSV.rows) {
        hashRow.push_back(new bool[32768]);
    }
    while(hashColumn.size() < captureFrameHSV.cols) {
        hashColumn.push_back(new bool[32768]);
    }
    for(bool*& x : hashRow) {
        for(int i=0; i<32768; i++) {
            x[i] = false;
        }
    }
    for(bool*& x : hashColumn) {
        for(int i=0; i<32768; i++) {
            x[i] = false;
        }
    }
    for(int i=0; i<captureFrameHSV.rows; i++) {
        for(int j=0; j<captureFrameHSV.cols; j++) {
            int c = smallColor(captureFrameHSV.at<Vec3b>(i,j));
            hashRow[i][c] = true;
            hashColumn[j][c] = true;
        }
    }
    int totalX = 0, totalY = 0, totalCorrectX = 0, totalCorrectY = 0;
    avgY = -1;
    int recogIndex = 0, prevj = -1;
    for(int j=0; j<captureFrameHSV.rows; j++) {
        int correct = 0;
        for(int k=0; k<recognizedImage.cols; k++) {
            int ind = smallColor(recognizedImage.at<Vec3b>(recogIndex,k));
            if(hashRow[j][ind] || (ind-1 >= 0 && hashRow[j][ind-1]) || (ind+1 < 32768 && hashRow[j][ind+1])
                    || (ind-32 >= 0 && hashRow[j][ind-32]) || (ind+32 < 32768 && hashRow[j][ind+32])
                    || (ind-1024 >= 0 && hashRow[j][ind-1024]) || (ind+1024 < 32768 && hashRow[j][ind+1024])
                    || (ind-2 >= 0 && hashColumn[j][ind-2]) || (ind+2 < 32768 && hashColumn[j][ind+2])
                    || (ind-64 >= 0 && hashColumn[j][ind-64]) || (ind+64 < 32768 && hashColumn[j][ind+64])
                    || (ind-2048 >= 0 && hashColumn[j][ind-2048]) || (ind+2048 < 32768 && hashColumn[j][ind+2048])) {
                correct++;
            }
        }
        if((float)correct/recognizedImage.cols > 0.94f) {
            totalY += j;
            totalCorrectY++;
            recogIndex++;
            prevj = j;
            if(recogIndex >= recognizedImage.rows) {
                avgY = totalY/totalCorrectY;
                break;
            }
        }
        if(j == captureFrameHSV.rows-1) {
            recogIndex++;
            if(recogIndex >= recognizedImage.rows)
                break;
            j = prevj;
        }
    }
    avgX = -1;
    recogIndex = 0;
    for(int j=0; j<captureFrameHSV.cols; j++) {
        int correct = 0;
        for(int k=0; k<recognizedImage.rows; k++) {
            int ind = smallColor(recognizedImage.at<Vec3b>(k,recogIndex));
            if(hashColumn[j][ind] || (ind-1 >= 0 && hashColumn[j][ind-1]) || (ind+1 < 32768 && hashColumn[j][ind+1])
                    || (ind-32 >= 0 && hashColumn[j][ind-32]) || (ind+32 < 32768 && hashColumn[j][ind+32])
                    || (ind-1024 >= 0 && hashColumn[j][ind-1024]) || (ind+1024 < 32768 && hashColumn[j][ind+1024])
                    || (ind-2 >= 0 && hashColumn[j][ind-2]) || (ind+2 < 32768 && hashColumn[j][ind+2])
                    || (ind-64 >= 0 && hashColumn[j][ind-64]) || (ind+64 < 32768 && hashColumn[j][ind+64])
                    || (ind-2048 >= 0 && hashColumn[j][ind-2048]) || (ind+2048 < 32768 && hashColumn[j][ind+2048])) {
                correct++;
            }
        }
        if((float)correct/recognizedImage.rows > 0.94f) {
            totalX += j;
            totalCorrectX++;
            recogIndex++;
            prevj = j;
            if(recogIndex >= recognizedImage.cols) {
                avgX = totalX/totalCorrectX;
                break;
            }
        }
        if(j == captureFrameHSV.cols-1) {
            recogIndex++;
            if(recogIndex >= recognizedImage.cols)
                break;
            j = prevj;
        }
    }
}

vector<pair<int,int>> getBorder(const Mat& m) {
    vector<pair<int,int>> b;
    for(int i=0; i<m.rows-1; i++) {
        for(int j=0; j<m.cols-1; j++) {
            if((m.at<Vec3b>(i,j)[0] < 128 && m.at<Vec3b>(i+1,j)[0] > 128) || (m.at<Vec3b>(i,j)[0] < 128 && m.at<Vec3b>(i,j+1)[0] > 128)
                    || (m.at<Vec3b>(i,j)[0] > 128 && m.at<Vec3b>(i+1,j)[0] < 128) || (m.at<Vec3b>(i,j)[0] > 128 && m.at<Vec3b>(i,j+1)[0] < 128)) {
                b.push_back(make_pair(i,j));
            }
        }
    }
    return b;
}

void myErode(const Mat& output, Mat& output2, int intensity) {
    bool neg = intensity < 0;
    intensity = abs(intensity);
    vector<pair<int,int>> borders = getBorder(output);
    for(const auto& p : borders) {
        int i = p.first;
        int j = p.second;
        if(output.at<Vec3b>(i,j)[0] > 128) {
            for(int a=-intensity+(neg?1:0); a<=intensity; a++) {
                for(int b=-intensity+(neg?1:0); b<=intensity; b++) {
                    int y = i+a;
                    int x = j+b;
                    if(y >= 0 && y < output.rows && x >= 0 && x < output.cols) {
                        output2.at<Vec3b>(y,x) = Vec3b(0,0,0);
                    }
                }
            }
        }
    }
}

void myDilate(const Mat& output2, Mat& output, int intensity) {
    vector<pair<int,int>> borders = getBorder(output2);
    for(const auto& p : borders) {
        int i = p.first;
        int j = p.second;
        if(output2.at<Vec3b>(i,j)[0] > 128) {
            for(int a=-intensity; a<=intensity; a++) {
                for(int b=-intensity; b<=intensity; b++) {
                    int y = i+a;
                    int x = j+b;
                    if(y >= 0 && y < output.rows && x >= 0 && x < output.cols) {
                        output.at<Vec3b>(y,x) = Vec3b(255,255,255);
                    }
                }
            }
        }
    }
}

vector<pair<int,pair<int,int>>>& getGraph(Mat& input, int intensity) {
    vector<pair<int,int>> borders = getBorder(input);
    bool pixel[input.rows][input.cols];

    vector<pair<int,pair<int,int>>> borderLinks;
    return borderLinks;
}

int colorDifference(const Vec3b& a, const Vec3b& b) {
    return (a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]) + (a[2]-b[2])*(a[2]-b[2]);
}

Mat bg;

Mat getNonBackground(Mat& frame) {
    static int first = 10;
    if(first-- > 0)
        bg = frame.clone();
    Mat output = frame.clone();
    for(int i=0; i<frame.rows; i++) {
        for(int j=0; j<frame.cols; j++) {
            output.at<Vec3b>(i,j) = Vec3b(255,255,255);
            for(int a=-1; a<=1; a++) {
                for(int b=-1; b<=1; b++) {
                    if(i+a >= 0 && j+b >= 0 && i+a < bg.rows && j+b < bg.cols) {
                        if(colorDifference(frame.at<Vec3b>(i,j), bg.at<Vec3b>(i+a,j+b)) < 1000) {
                            output.at<Vec3b>(i,j) = Vec3b(0,0,0);
                            break;
                        }
                    }
                }
            }
        }
    }
    return output;
}

vector<Vec3b> colors;
Mat bg1,bg2;

int dir[][2] = {{-1,0},{0,1},{1,0},{0,-1}};

vector<pair<Point,Point>> myConnectedComponents(Mat& img, Mat& out, Mat& bg) {
    out = img.clone();
    int visited[img.rows][img.cols];
    vector<pair<Point,Point>> bound;
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            visited[i][j] = -1;
        }
    }
    queue<pair<int,int>> q;
    int cnt = 0;
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(visited[i][j] == -1 && img.at<Vec3b>(i,j)[0] > 128 && bg.at<Vec3b>(i,j)[0] > 128) {
                visited[i][j] = cnt++;
                bound.push_back(make_pair(Point(img.cols,img.rows),Point(0,0)));
                q.push(make_pair(i,j));
                if(colors.size() < cnt)
                    colors.push_back(Vec3b((int)(rand()*255),(int)(rand()*255),(int)(rand()*255)));
            }
            while(!q.empty()) {
                pair<int,int> p = q.front();
                q.pop();
                out.at<Vec3b>(p.first,p.second) = colors[visited[p.first][p.second]];
                bound[visited[p.first][p.second]].first.x = min(bound[visited[p.first][p.second]].first.x, p.second);
                bound[visited[p.first][p.second]].first.y = min(bound[visited[p.first][p.second]].first.y, p.first);
                bound[visited[p.first][p.second]].second.x = max(bound[visited[p.first][p.second]].second.x, p.second);
                bound[visited[p.first][p.second]].second.y = max(bound[visited[p.first][p.second]].second.y, p.first);
                for(int d=0; d<4; d++) {
                    int y = p.first + dir[d][0];
                    int x = p.second + dir[d][1];
                    if(y >= 0 && x >= 0 && y < img.rows && x < img.cols) {
                        if(visited[y][x] == -1 && img.at<Vec3b>(y,x)[0] > 128 && bg.at<Vec3b>(y,x)[0] > 128) {
                            visited[y][x] = visited[i][j];
                            q.push(make_pair(y,x));
                        }
                    }
                }
            }
        }
    }
    for(auto& x : bound) {
        x.first.x *= 2;
        x.first.y *= 2;
        x.second.x *= 2;
        x.second.y *= 2;
    }
    return bound;
}


Point getBiggestComponent(Mat& input, vector<pair<Point,Point>>& bounds, Mat& output) {
    int index = -1;
    int maxArea = 0;
    for(int i=0; i<bounds.size(); i++) {
        int area = (bounds[i].second.x - bounds[i].first.x) * (bounds[i].second.y - bounds[i].first.y);
        if(area > maxArea) {
            maxArea = area;
            index = i;
        }
    }
    if(index==-1) {
        output = Mat(1,1,CV_8UC3,Vec3b(0,0,0));
        return Point(-1,-1);
    }
    output = Mat((bounds[index].second.y - bounds[index].first.y)/2,(bounds[index].second.x - bounds[index].first.x)/2, CV_8UC3, Vec3b(0,0,0));
    for(int i=max(0,bounds[index].first.y/2); i<min(input.rows,bounds[index].second.y/2); i++) {
        for(int j=max(0,bounds[index].first.x/2); j<min(input.cols,bounds[index].second.x/2); j++) {
            if(input.at<Vec3b>(i,j) == colors[index]) {
                output.at<Vec3b>(i-bounds[index].first.y/2, j-bounds[index].first.x/2) = Vec3b(255,255,255);
            }
        }
    }
    return Point(bounds[index].first.x,bounds[index].first.y);
}

vector<pair<Point,Point>> bounds;

void fillHole(Mat& in, Mat& out2) {
    Mat out = Mat(in.rows,in.cols,CV_8UC3,Vec3b(255,255,255));
    bool visited[in.rows][in.cols];
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            visited[i][j] = false;
        }
    }
    queue<pair<int,int>> q;
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            if(i!=0 && i!=in.rows-1) {
                if(j!=0) {
                    j = in.cols-1;
                }
            }
            if(!visited[i][j] && in.at<Vec3b>(i,j)[0] < 128) {
                visited[i][j] = true;
                q.push(make_pair(i,j));
            }
            while(!q.empty()) {
                pair<int,int> p = q.front();
                q.pop();
                out.at<Vec3b>(p.first,p.second) = Vec3b(0,0,0);
                for(int d=0; d<4; d++) {
                    int y = p.first + dir[d][0];
                    int x = p.second + dir[d][1];
                    if(y >= 0 && x >= 0 && y < in.rows && x < in.cols) {
                        if(!visited[y][x] && in.at<Vec3b>(y,x)[0] < 128) {
                            visited[y][x] = true;
                            q.push(make_pair(y,x));
                        }
                    }
                }
            }
        }
    }
    out2 = out;
}

int disjointParent(int *disjoint, int a) {
    if(disjoint[a] != a) {
        disjoint[a] = disjointParent(disjoint, disjoint[a]);
    }
    return disjoint[a];
}
void disjointMerge(int *disjoint, int a, int b) {
    int mini = min(disjointParent(disjoint,a), disjointParent(disjoint,b));
    int maxi = max(disjointParent(disjoint,a), disjointParent(disjoint,b));
    disjoint[maxi] = mini;
}

void connectDot(const Mat& bg, const Mat& in, Mat& out, int dist) {
    out = in.clone();
    int visited[in.rows][in.cols];
    int visited2[dist*2+1][dist*2+1];
    queue<Point> q;
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            visited[i][j] = i*in.cols+j;
        }
    }
    bool done = false;
    while(!done) {
        for(int i=0; i<in.rows; i++) {
            for(int j=0; j<in.cols; j++) {
                if(in.at<Vec3b>(i,j)[0] > 128) {
                    q.push(Point(j,i));
                }
            }
        }
        done = true;
        while(!q.empty()) {
            Point p = q.front();
            q.pop();
            for(int i=0; i<dist*2+1; i++) {
                for(int j=0; j<dist*2+1; j++) {
                    visited2[i][j] = dist*2+1;
                }
            }
            visited2[dist][dist] = 0;
            queue<Point> q2;
            q2.push(Point(p.x,p.y));
            while(!q2.empty()) {
                Point p2 = q2.front();
                q2.pop();
                if(disjointParent(*visited,visited[p2.y][p2.x]) != disjointParent(*visited,visited[p.y][p.x]) && in.at<Vec3b>(p2.y,p2.x)[0] > 128) {
                    done = false;
                    while(visited2[dist-p.y+p2.y][dist-p.x+p2.x] >= 0) {
                        disjointMerge(*visited,visited[p2.y][p2.x],visited[p.y][p.x]);
                        out.at<Vec3b>(p2.y,p2.x) = Vec3b(255,255,255);
                        bool found = false;
                        for(int d=0; d<4; d++) {
                            int y = p2.y+dir[d][0];
                            int x = p2.x+dir[d][1];
                            if(x >= 0 && y >= 0 && x < in.cols && y < in.rows && visited2[dist-p.y+p2.y][dist-p.x+p2.x] > visited2[dist-p.y+y][dist-p.x+x]) {
                                p2.y = y;
                                p2.x = x;
                                found = true;
                                break;
                            }
                        }
                        if(!found) {
                            break;
                        }
                    }
                    break;
                } else if(visited2[dist-p.y+p2.y][dist-p.x+p2.x] >= dist) {
                    break;
                }
                for(int d=0; d<4; d++) {
                    int y = p2.y+dir[d][0];
                    int x = p2.x+dir[d][1];
                    if(x >= 0 && y >= 0 && x < in.cols && y < in.rows && visited2[dist-p.y+y][dist-p.x+x] > 2*dist && visited2[dist-p.y+p2.y][dist-p.x+p2.x]+1 < visited2[dist-p.y+y][dist-p.x+x] && bg.at<Vec3b>(y,x)[0] > 128) {
                        q2.push(Point(x,y));
                        visited2[dist-p.y+y][dist-p.x+x] = visited2[dist-p.y+p2.y][dist-p.x+p2.x]+1;
                    }
                }
            }
        }
    }
}

void thinLineFilter(const Mat& in, Mat& out) {
    for(int i=1; i<in.rows-1; i++) {
        for(int j=1; j<in.cols-1; j++) {
            if((in.at<Vec3b>(i-1,j)[0] < 128 && in.at<Vec3b>(i+1,j)[0] < 128) ||
                    (in.at<Vec3b>(i,j-1)[0] < 128 && in.at<Vec3b>(i,j+1)[0] < 128)) {
                out.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }else{
                out.at<Vec3b>(i,j) = in.at<Vec3b>(i,j);
            }
        }
    }
}

void dotFilter(const Mat& in, Mat& out) {
    for(int i=1; i<in.rows-1; i++) {
        for(int j=1; j<in.cols-1; j++) {
            if((in.at<Vec3b>(i-1,j)[0] < 128 && in.at<Vec3b>(i+1,j)[0] < 128) &&
                    (in.at<Vec3b>(i,j-1)[0] < 128 && in.at<Vec3b>(i,j+1)[0] < 128)) {
                out.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }else{
                out.at<Vec3b>(i,j) = in.at<Vec3b>(i,j);
            }
        }
    }
}

pair<Point,int> biggestCenter(Mat& in, Mat& out2, Mat& out3) {
    Mat out = Mat(in.rows,in.cols,CV_8UC3,Vec3b(255,255,255));
    int visited[in.rows][in.cols];
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            visited[i][j] = -1;
        }
    }
    queue<pair<int,int>> q;
    queue<pair<int,int>> q2;
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            if(i!=0 && i!=in.rows-1) {
                if(j!=0) {
                    j = in.cols-1;
                }
            }
            if(visited[i][j] == -1 && in.at<Vec3b>(i,j)[0] < 128) {
                visited[i][j] = 0;
                q.push(make_pair(i,j));
            }
        }
    }
    int maxv = 1;
    while(!q.empty()) {
        pair<int,int> p = q.front();
        q.pop();
        for(int d=0; d<4; d++) {
            int y = p.first + dir[d][0];
            int x = p.second + dir[d][1];
            if(y >= 0 && x >= 0 && y < in.rows && x < in.cols) {
                if(visited[y][x] == -1) {
                    if(in.at<Vec3b>(y,x)[0] < 128) {
                        visited[y][x] = visited[p.first][p.second];
                        q.push(make_pair(y,x));
                    } else if(in.at<Vec3b>(y,x)[0] > 128) {
                        visited[y][x] = visited[p.first][p.second]+1;
                        q2.push(make_pair(y,x));
                    }
                }
            }
        }
    }
    bool out3temp[in.rows][in.cols];
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            out3temp[i][j] = false;
        }
    }
    while(!q2.empty()) {
        pair<int,int> p = q2.front();
        q2.pop();
        bool found = false;
        for(int d=0; d<4; d++) {
            int y = p.first + dir[d][0];
            int x = p.second + dir[d][1];
            if(y >= 0 && x >= 0 && y < in.rows && x < in.cols) {
                if(visited[y][x] == -1 || visited[y][x] > visited[p.first][p.second] + 1) {
                    visited[y][x] = visited[p.first][p.second] + 1;
                    q2.push(make_pair(y,x));
                    found = true;
                }
            }
        }
        if(!found) {
            out3temp[p.first][p.second] = true;
        }
    }
    Point ret;
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            if(visited[i][j] > maxv) {
                maxv = visited[i][j];
                ret = Point(j*2,i*2);
            }
        }
    }
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            int val = (visited[i][j] > 0 ? 50 : 0) + 205 * visited[i][j] / maxv;
            out.at<Vec3b>(i,j) = Vec3b(val,val,val);
        }
    }
    out2 = out.clone();
    out3 = out.clone();
    for(int i=0; i<in.rows; i++) {
        for(int j=0; j<in.cols; j++) {
            if(out3temp[i][j]) {
                if(visited[i][j] < maxv/3) {
                    out3.at<Vec3b>(i,j) = Vec3b(255,255,255);
                } else {
                    out3.at<Vec3b>(i,j) = Vec3b(0,0,0);
                }
                out3temp[i][j] = false;
            } else {
                out3.at<Vec3b>(i,j) = Vec3b(0,0,0);
            }
        }
    }
    thinLineFilter(out3.clone(), out3);
    dotFilter(out3.clone(), out3);
    connectDot(in.clone(), out3.clone(), out3, 10);
    return make_pair(ret,maxv*2);
}

double pointLength(const pair<Point,Point>& a){
    return hypot(a.first.x-a.second.x,a.first.y-a.second.y);
}

double pointAngle(const Point& a, const Point& b){
    return atan2(b.x-a.x,b.y-a.y);
}

double angleAngle(const Point& a, const Point& b, const Point& c){
    double diff = abs(pointAngle(a,c)-pointAngle(b,c));
    return min(diff, 2*M_PI-diff);
}

bool lengthCmp(const pair<Point,Point>& a, const pair<Point,Point>& b){
    return pointLength(a) < pointLength(b);
}

struct AngleCmp {
    Point center;
    AngleCmp(Point center){
        this->center = center;
    }
    bool operator()(const pair<Point,Point>& a, const pair<Point,Point>& b) {
        return pointAngle(a.second,center) > pointAngle(b.second,center);
    }
};

vector<pair<Point,Point>> getHand(const Mat& img, const pair<Point,int>& center) {
    vector<pair<Point,Point>> out;
    int visited[img.rows][img.cols];
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            visited[i][j] = -1;
        }
    }
    int cnt = 0;
    queue<pair<int,int>> q;
    for(int i=0; i<img.rows; i++) {
        for(int j=0; j<img.cols; j++) {
            if(visited[i][j] == -1 && img.at<Vec3b>(i,j)[0] > 128) {
                visited[i][j] = cnt++;
                q.push(make_pair(i,j));
                Point fartest = Point(center.first.x/2,center.first.y/2);
                Point nearest = Point(img.cols*2,img.rows*2);
                while(!q.empty()) {
                    pair<int,int> p = q.front();
                    q.pop();
                    if(lengthCmp(make_pair(fartest,center.first/2), make_pair(Point(p.second,p.first),center.first/2))){
                        fartest = Point(p.second,p.first);
                    }
                    if(lengthCmp(make_pair(Point(p.second,p.first),center.first/2), make_pair(nearest,center.first/2))){
                        nearest = Point(p.second,p.first);
                    }
                    for(int d=0; d<4; d++) {
                        int y = p.first + dir[d][0];
                        int x = p.second + dir[d][1];
                        if(y >= 0 && x >= 0 && y < img.rows && x < img.cols) {
                            if(visited[y][x] == -1 && img.at<Vec3b>(y,x)[0] > 128) {
                                visited[y][x] = visited[i][j];
                                q.push(make_pair(y,x));
                            }
                        }
                    }
                }
                out.push_back(make_pair(fartest*2,nearest*2));
            }
        }
    }
    sort(out.begin(),out.end(),lengthCmp);
    out = vector<pair<Point,Point>>(out.size()>5?out.end()-5:out.begin(), out.end());
    sort(out.begin(),out.end(),AngleCmp(center.first));
    int thumb = 0, maxdist = 0;
    for(int i=0;i<out.size();i++){
        int dist = 0;
        for(int j=0;j<out.size();j++){
            dist += pointLength(make_pair(out[i].second,out[j].second));
        }
        if(dist > maxdist){
            maxdist = dist;
            thumb = i;
        }
    }
    if(out.size() > 0 && angleAngle(out[thumb].second,out[(thumb-1+out.size())%out.size()].second,center.first) <
       angleAngle(out[thumb].second,out[(thumb+1)%out.size()].second,center.first)) {
        reverse(out.begin(),out.end());
        thumb = out.size()-thumb-1;
    }
    out.insert(out.end(),out.begin(),out.end());
    return vector<pair<Point,Point>>(out.begin()+thumb,out.begin()+thumb+out.size()/2);
}

Mat bigOutputSkeleton;
Point bigOutputPos;
pair<Point,int> bigOutputSkeletonPos;
vector<pair<Point,Point>> outputHandPos;

void initialize_color() {
    maxb.push_back(Vec3b(120,125,164));
    minb.push_back(Vec3b(106,50,119));
    maxb.push_back(Vec3b(121,130,180));
    minb.push_back(Vec3b(108,73,110));
    maxb.push_back(Vec3b(122,117,167));
    minb.push_back(Vec3b(106,68,119));
    maxb.push_back(Vec3b(121,128,138));
    minb.push_back(Vec3b(106,74,104));
    maxb.push_back(Vec3b(120,119,138));
    minb.push_back(Vec3b(108,75,98));
    maxb.push_back(Vec3b(120,139,150));
    minb.push_back(Vec3b(106,83,96));
    maxb.push_back(Vec3b(119,150,164));
    minb.push_back(Vec3b(108,77,108));
    maxb.push_back(Vec3b(121,111,222));
    minb.push_back(Vec3b(107,50,129));
    maxb.push_back(Vec3b(121,114,150));
    minb.push_back(Vec3b(105,52,117));
    maxb.push_back(Vec3b(113,86,225));
    minb.push_back(Vec3b(102,58,156));
    maxb.push_back(Vec3b(112,94,223));
    minb.push_back(Vec3b(104,64,145));
    maxb.push_back(Vec3b(119,107,173));
    minb.push_back(Vec3b(101,49,128));
    maxb.push_back(Vec3b(119,138,249));
    minb.push_back(Vec3b(107,62,141));
    maxb.push_back(Vec3b(123,119,174));
    minb.push_back(Vec3b(108,40,133));
    maxb.push_back(Vec3b(122,101,198));
    minb.push_back(Vec3b(110,49,149));
    maxb.push_back(Vec3b(122,114,205));
    minb.push_back(Vec3b(110,31,151));
    maxb.push_back(Vec3b(112,90,185));
    minb.push_back(Vec3b(99,53,152));
    maxb.push_back(Vec3b(112,93,185));
    minb.push_back(Vec3b(100,55,159));
    maxb.push_back(Vec3b(113,118,166));
    minb.push_back(Vec3b(91,52,128));
    maxb.push_back(Vec3b(118,115,156));
    minb.push_back(Vec3b(102,63,120));
    maxb.push_back(Vec3b(119,124,192));
    minb.push_back(Vec3b(109,74,119));
    maxb.push_back(Vec3b(117,142,188));
    minb.push_back(Vec3b(110,90,98));
    maxb.push_back(Vec3b(124,124,162));
    minb.push_back(Vec3b(108,73,99));
    maxb.push_back(Vec3b(115,113,255));
    minb.push_back(Vec3b(103,56,153));
    maxb.push_back(Vec3b(113,123,138));
    minb.push_back(Vec3b(101,72,109));
    maxb.push_back(Vec3b(111,126,139));
    minb.push_back(Vec3b(101,80,107));
    maxb.push_back(Vec3b(112,111,118));
    minb.push_back(Vec3b(102,79,95));
    maxb.push_back(Vec3b(111,123,147));
    minb.push_back(Vec3b(100,70,107));
    maxb.push_back(Vec3b(113,136,130));
    minb.push_back(Vec3b(101,86,96));
    maxb.push_back(Vec3b(112,109,181));
    minb.push_back(Vec3b(93,56,131));
    maxb.push_back(Vec3b(111,94,158));
    minb.push_back(Vec3b(101,55,142));
    maxb.push_back(Vec3b(110,104,191));
    minb.push_back(Vec3b(101,61,148));
    maxb.push_back(Vec3b(106,93,241));
    minb.push_back(Vec3b(100,61,207));
    maxb.push_back(Vec3b(107,91,241));
    minb.push_back(Vec3b(101,64,213));
    maxb.push_back(Vec3b(111,126,184));
    minb.push_back(Vec3b(100,77,132));
    maxb.push_back(Vec3b(111,118,253));
    minb.push_back(Vec3b(100,58,123));
    maxb.push_back(Vec3b(121,168,152));
    minb.push_back(Vec3b(104,75,73));
    maxb.push_back(Vec3b(115,108,158));
    minb.push_back(Vec3b(94,50,109));
    maxb.push_back(Vec3b(117,123,206));
    minb.push_back(Vec3b(98,53,111));
}
void initialize_color_dark() {
    maxb.push_back(Vec3b(177,33,255));
    minb.push_back(Vec3b(0,0,207));
    maxb.push_back(Vec3b(121,55,191));
    minb.push_back(Vec3b(109,30,166));
    maxb.push_back(Vec3b(130,52,202));
    minb.push_back(Vec3b(109,21,165));
    maxb.push_back(Vec3b(170,54,225));
    minb.push_back(Vec3b(0,1,173));
    maxb.push_back(Vec3b(177,39,204));
    minb.push_back(Vec3b(0,3,134));
    maxb.push_back(Vec3b(177,46,206));
    minb.push_back(Vec3b(9,3,128));
    maxb.push_back(Vec3b(170,66,222));
    minb.push_back(Vec3b(24,3,97));
    maxb.push_back(Vec3b(176,46,153));
    minb.push_back(Vec3b(0,0,96));
    maxb.push_back(Vec3b(178,34,212));
    minb.push_back(Vec3b(0,0,158));
    maxb.push_back(Vec3b(177,49,158));
    minb.push_back(Vec3b(0,0,124));
    maxb.push_back(Vec3b(177,100,155));
    minb.push_back(Vec3b(0,0,89));
    maxb.push_back(Vec3b(90,36,151));
    minb.push_back(Vec3b(17,5,103));
    maxb.push_back(Vec3b(179,36,232));
    minb.push_back(Vec3b(0,2,169));
    maxb.push_back(Vec3b(179,30,236));
    minb.push_back(Vec3b(0,0,165));
    maxb.push_back(Vec3b(179,36,253));
    minb.push_back(Vec3b(0,2,88));
    maxb.push_back(Vec3b(179,39,254));
    minb.push_back(Vec3b(0,2,136));
    maxb.push_back(Vec3b(177,49,230));
    minb.push_back(Vec3b(0,0,180));
    maxb.push_back(Vec3b(110,45,205));
    minb.push_back(Vec3b(13,3,163));
    maxb.push_back(Vec3b(178,54,174));
    minb.push_back(Vec3b(1,21,104));
}

void auto_initialize_color_hand(){
    maxb.clear();
    minb.clear();

    maxb.push_back(Vec3b(114,141,172));
    minb.push_back(Vec3b(108,71,117));
    maxb.push_back(Vec3b(114,166,193));
    minb.push_back(Vec3b(108,96,112));
    maxb.push_back(Vec3b(113,182,168));
    minb.push_back(Vec3b(108,99,122));
    maxb.push_back(Vec3b(112,140,211));
    minb.push_back(Vec3b(108,72,142));
    maxb.push_back(Vec3b(115,231,115));
    minb.push_back(Vec3b(108,84,55));
    maxb.push_back(Vec3b(115,143,186));
    minb.push_back(Vec3b(108,72,102));
    maxb.push_back(Vec3b(113,174,160));
    minb.push_back(Vec3b(108,88,97));
    maxb.push_back(Vec3b(116,157,228));
    minb.push_back(Vec3b(108,57,107));
}

bool standby = false;
int fnFrameCounter[] = {0,0,0,0,0,0,0,0,0};
vector<int> fnFrameCountBuffer;
vector<bool> isRealFinger = {false,false,false,false,false};
int real_finger_count;
VideoCapture cap = VideoCapture(0);
double fps;
Point palmPos;
bool isPalmPosMarked = false;

void calculateFPS(){
    /* Frame Rate */
    //double fps = cap.get(CV_CAP_PROP_FPS);
    // If you do not care about backward compatibility
    // You can use the following instead for OpenCV 3
    fps = cap.get(CAP_PROP_FPS);
    cout << "Frames per second using cap.get(CV_CAP_PROP_FPS) : " << fps << endl;
    // Number of frames to capture
    int num_frames = 120;
    // Start and end times
    time_t start, end;
    // Variable for storing video frames
    Mat frame;
    cout << "Capturing " << num_frames << " frames" << endl ;
    // Start time
    time(&start);
    // Grab a few frames
    for(int i = 0; i < num_frames; i++){
        cap >> frame;
    }
    // End Time
    time(&end);
    // Time elapsed
    double seconds = difftime (end, start);
    cout << "Time taken : " << seconds << " seconds" << endl;
    // Calculate frames per second
    fps  = num_frames / seconds;
    cout << "Estimated frames per second : " << fps << endl;
}

int findMode(vector<int> v){
    vector<int> counter (9,0);
    for(int i=0; i<v.size(); i++){counter[v[i]]++;}
    return max_element(counter.begin(), counter.end()) - counter.begin();
}

void startTrackPalm(){
    palmPos = bigOutputSkeletonPos.first+bigOutputPos;
    isPalmPosMarked = true;
}

void stopTrackPalm(){
    isPalmPosMarked = false;
}

double getTrackedPalmDistance(){
    if(isPalmPosMarked){
        Point nowPalmPos = (bigOutputSkeletonPos.first+bigOutputPos);
        return hypot(palmPos.x-nowPalmPos.x,palmPos.y-nowPalmPos.y);
    } else {return 0;}
}

//Windows Key
INPUT ip;
void initWindowsKey(){
    ip.type = INPUT_KEYBOARD;
    ip.ki.wScan = 0; // hardware scan code for key
    ip.ki.time = 0;
    ip.ki.dwExtraInfo = 0;
}
void press(int code){
    // Press the "A" key
    ip.ki.wVk = code; // virtual-key code for the "a" key
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));

    // Release the "A" key
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
}
void pressHex(string code){
    unsigned int x;
    std::stringstream ss;
    ss << std::hex << code;
    ss >> x;
    press(x);
    return;
}

string getActiveWindowTitle(){
    char wnd_title[256];
    HWND hwnd = GetForegroundWindow(); // get handle of currently active window
    GetWindowText(hwnd,wnd_title,sizeof(wnd_title));
    return wnd_title;
}

bool isActiveWindowYoutube(){
    string title = getActiveWindowTitle();
    if(title.find("YouTube") == -1) return false;
    else return true;
}

bool isInTaskView(){
    string title = getActiveWindowTitle();
    if(title.find("Task View") == -1) return false;
    else return true;
}

bool isInSlideShow(){
    string title = getActiveWindowTitle();
    if(title.find("Slide Show") == -1) return false;
    else return true;
}

bool isInPowerPoint(){
    string title = getActiveWindowTitle();
    if(title.find("PowerPoint") == -1) return false;
    else return true;
}

void enterTaskView(){
    ip.ki.wVk = 91; //press win
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));

    ip.ki.wVk = 9; //press tab
    ip.ki.dwFlags = 0; // 0 for key press
    SendInput(1, &ip, sizeof(INPUT));

    ip.ki.wVk = 9; //release tab
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));

    ip.ki.wVk = 91; //release win
    ip.ki.dwFlags = KEYEVENTF_KEYUP; // KEYEVENTF_KEYUP for key release
    SendInput(1, &ip, sizeof(INPUT));
}

float getFingerAngle(pair<Point,Point> fingerPos){
    int baseX, baseY, tipX, tipY;
    tipX = fingerPos.first.x;
    tipY = fingerPos.first.y;
    baseX = fingerPos.second.x;
    baseY = fingerPos.second.y;
    float uncalibrateAngle = atan2(tipY-baseY,tipX-baseX) * 180 / M_PI;
    if(uncalibrateAngle > -180 && uncalibrateAngle <= 90){
        return uncalibrateAngle + 90;
    } else if(uncalibrateAngle > 90 && uncalibrateAngle <= 180){
        return uncalibrateAngle - 270;
    }
}

float getAvgFingersAngleByIndex(int startIndex, int endIndex, vector<pair<Point,Point>> fingersPos) {
    float angle = 0;
    float weightSum = 0;
    for(int i=startIndex; i<=endIndex; i++){
        angle+=getFingerAngle(fingersPos[i]) * pointLength(fingersPos[i]);
        weightSum+=pointLength(fingersPos[i]);
    }
    return angle/weightSum;
}

float getAvgFingersAngle(vector<pair<Point,Point>> fingersPos){
    if(real_finger_count==0) return 0;

    float angle = 0;
    float weightSum = 0;
    for(int i=0; i<real_finger_count; i++){
        if(!isRealFinger[i]) continue;
        angle+=getFingerAngle(fingersPos[i]) * pointLength(fingersPos[i]);
        weightSum+=pointLength(fingersPos[i]);
    }
    return angle/weightSum;
}

void clearOtherFnFrameCount(int fnNum){
    for(int i=0; i<8; i++){
        if(i==fnNum) continue;
        else fnFrameCounter[i] = 0;
    }
}

void clearAllFnFrameCount(){
    for(int i=0; i<8; i++)
        fnFrameCounter[i] = 0;
}

void triggerFunction(int fnNum){
    //implement actual function here
    bool isYoutube = isActiveWindowYoutube();

    switch(fnNum){
        case 0:
            //Play/Pause
            if (standby) {
                cout<<"Play/Pause"<<endl;
                if(isYoutube) pressHex("0x20"); //Youtube
                else pressHex("0xB3");
            }
            standby = false;
            break;
        case 1:
            //Mute/Unmute
            if (standby) {
                cout<<"Mute/Unmute"<<endl;
                pressHex("0xAD");
            }
            standby = false;
            break;
        case 2:
            if (standby) {
                cout<<"next"<<endl;
                if(isYoutube) pressHex("0x27"); //Youtube
                else pressHex("0xB0");
            }
            standby = false;
            break;
        case 3:
            if (standby) {
                cout<<"prevoius"<<endl;
                if(isYoutube) pressHex("0x25"); //Youtube
                else pressHex("0xB1");
            }
            standby = false;
            break;
        case 4:
            cout<<"volumn up"<<endl;
            if(isYoutube) pressHex("0x26"); //Youtube
            else pressHex("0xAF");
            break;
        case 5:
            cout<<"volumn down"<<endl;
            if(isYoutube) pressHex("0x28"); //Youtube
            else pressHex("0xAE");
            break;
        case 6:
            //fullscreen
            cout<<"full screen"<<endl;
            if(isYoutube) pressHex("0x46"); //Youtube
            else pressHex("0x7A");
            clearAllFnFrameCount();
            stopTrackPalm();
            break;
        case 7:
            //exit fullscreen
            cout<<"exit full screen"<<endl;
            if(isYoutube) pressHex("0x1B"); //Youtube
            else pressHex("0x7A");
            clearAllFnFrameCount();
            stopTrackPalm();
            break;
        case 8:
            //standby
            cout<<"standby"<<endl;
            fnFrameCounter[0] = 0;
            fnFrameCounter[1] = 0;
            fnFrameCounter[2] = 0;
            fnFrameCounter[3] = 0;
            standby = true;
            break;
        default:
            cout << "function " << fnNum << " triggered" << endl;
            break;
    }
}

void checkFnTrigger(int fnNum){
    //adjust number of frame to count before trigger function
    int frameCountRequireToTrigger[] = {10,10,5,5,5,5,10,10,15}; //last index of array is standby post
    if(fnFrameCounter[fnNum] >= frameCountRequireToTrigger[fnNum]){
        fnFrameCounter[fnNum] = 0;
        triggerFunction(fnNum);
    }
}

//call this when the frame meets the required condition
int BUFFER = 7;
void countFnFrame(int fnNum){
    //clearOtherFnFrameCount(fnNum);

    //for remove noise frame
    if(fnFrameCountBuffer.size() < BUFFER){
        fnFrameCountBuffer.insert(fnFrameCountBuffer.begin(), fnNum);
        return;
    } else {
        fnFrameCountBuffer.pop_back();
        fnFrameCountBuffer.insert(fnFrameCountBuffer.begin(), fnNum);
        int mode = findMode(fnFrameCountBuffer);
        fnFrameCounter[mode]++;
        checkFnTrigger(mode);
    }
}

void clearBuffer(){
    fnFrameCountBuffer.clear();
}

void showRealFingerCount(Mat frame, int c){
    putText	(
        frame,
        "Real Finger: "+to_string(c),
        Point((int)frame.cols/2, (int)frame.rows*0.89),
        FONT_HERSHEY_PLAIN,
        1,
        Scalar(255,255,255)
    );
}
void showAngel(Mat frame){
    putText	(
        frame,
        "Angle: "+to_string(getAvgFingersAngle(outputHandPos)),
        Point((int)frame.cols/2, (int)frame.rows*0.92),
        FONT_HERSHEY_PLAIN,
        1,
        Scalar(255,255,255)
    );
}
void showBuffer(Mat frame){
    string s;
    for(int i=0; i<fnFrameCountBuffer.size(); i++){
        s+=to_string(fnFrameCountBuffer[i])+" ";
    }
    putText	(
        frame,
        "buffer: "+s,
        Point((int)frame.cols/2, (int)frame.rows*0.95),
        FONT_HERSHEY_PLAIN,
        1,
        Scalar(255,255,255)
    );
}
void showMode(Mat frame){
    putText	(
        frame,
        "Mode: "+to_string(findMode(fnFrameCountBuffer)),
        Point((int)frame.cols/2, (int)frame.rows*0.98),
        FONT_HERSHEY_PLAIN,
        1,
        Scalar(255,255,255)
    );
}
void showAllFingerAngle(Mat frame){
    float fingerAngle;
    for(int i=0; i<5; i++){
        if(!isRealFinger[i]) continue;
        fingerAngle = getFingerAngle(outputHandPos[i]);
        putText	(
            frame,
            to_string(fingerAngle),
            outputHandPos[i].first+bigOutputPos,
            FONT_HERSHEY_PLAIN,
            1,
            Scalar(255,255,255)
        );
    }
}
void showPalmTrack(Mat frame){
    if(isPalmPosMarked){
        Point nowPalmPos = (bigOutputSkeletonPos.first+bigOutputPos);
        float palmLineAngle = getFingerAngle(make_pair(nowPalmPos, palmPos));
        circle(frame,palmPos,1,Scalar(255,222,0),10);
        line(shownCaptureFrame,palmPos,nowPalmPos, Scalar(255,246,184), 5);
        putText	(
            frame,
            to_string(getTrackedPalmDistance()) + " | " + to_string(palmLineAngle) + " degree",
            (nowPalmPos+palmPos)/2,
            FONT_HERSHEY_PLAIN,
            1,
            Scalar(255,255,255)
        );
    }
}

int main() {
    initWindowsKey();

    initialize_color();
    auto_initialize_color_hand();

    if(!cap.isOpened()) {
        cout << "Failed" << endl;
        return -1;
    }
    namedWindow("Webcam");
    namedWindow("Output");
    setMouseCallback("Webcam", myMouseCallback, NULL);

    //calculateFPS();
    int cooldown = 0;

    while(true) {

        cap >> captureFrameOriginal;
        captureFrame = Mat(captureFrameOriginal.rows,captureFrameOriginal.cols,CV_8UC3);
        if(captureFrame.data) {
            for(int i=0; i<captureFrame.rows; i++) {
                for(int j=0; j<captureFrame.cols; j++) {
                    captureFrame.at<Vec3b>(i,captureFrame.cols-j-1) = captureFrameOriginal.at<Vec3b>(i,j);
                }
            }
            captureFrameOriginal = captureFrame.clone();
            resize(captureFrame,captureFrame,Size(320,240));
            cvtColor(captureFrame,captureFrameHSV,CV_RGB2HSV);
            shownCaptureFrame = captureFrameOriginal.clone();
            if(pos[0] > -1 && pos[1] > -1) {
                for(int i=pos[1]; i<cpos[1]; i++) {
                    shownCaptureFrame.at<Vec3b>(pos[0],i) = Vec3b(255,255,255);
                    shownCaptureFrame.at<Vec3b>(cpos[0],i) = Vec3b(255,255,255);
                }
                for(int i=pos[0]; i<cpos[0]; i++) {
                    shownCaptureFrame.at<Vec3b>(i,pos[1]) = Vec3b(255,255,255);
                    shownCaptureFrame.at<Vec3b>(i,cpos[1]) = Vec3b(255,255,255);
                }
            }
            //for(int i=0; i<bounds.size(); i++) {
            //    rectangle(shownCaptureFrame,Rect(bounds[i].first,bounds[i].second),colors[i],3);
            //}

            circle(shownCaptureFrame,bigOutputSkeletonPos.first+bigOutputPos,1,Scalar(100,0,0),10);
            Vec3b fingerColors[] = {Vec3b(235,0,200),Vec3b(255,100,0),Vec3b(0,200,0),Vec3b(0,180,200),Vec3b(0,50,220)};
            Vec3b white = Vec3b(255,255,255);
            Vec3b tip = Vec3b(73,195,90);
            Vec3b base = Vec3b(136,150,0);
            Vec3b red = Vec3b(0,0,255);

            //finger_count
            real_finger_count = 0;
            for(int i=0;i<outputHandPos.size();i++){
                isRealFinger[i] = false;
                if (cv::norm((outputHandPos[i].first+bigOutputPos)-(outputHandPos[i].second+bigOutputPos)) > 50){
                    real_finger_count++;
                    isRealFinger[i] = true;
                }
            }
            showRealFingerCount(shownCaptureFrame, real_finger_count);

            for(int i=0;i<outputHandPos.size();i++){
                if(isRealFinger[i]){
                    line(shownCaptureFrame,outputHandPos[i].first+bigOutputPos,outputHandPos[i].second+bigOutputPos, white, 5);
                    circle(shownCaptureFrame,outputHandPos[i].second+bigOutputPos,4, base, 5);
                    circle(shownCaptureFrame,outputHandPos[i].first+bigOutputPos,4, tip, 5);
                } else {
                    line(shownCaptureFrame,outputHandPos[i].first+bigOutputPos,outputHandPos[i].second+bigOutputPos, red, 5);
                }
            }

            //finger_count
            if(real_finger_count==0) {
                standby = true;
                clearBuffer();
                stopTrackPalm();
            }

            if(real_finger_count==1) {
                stopTrackPalm();
            }

            if (real_finger_count==2) {
                float avgAng = getAvgFingersAngle(outputHandPos);
                if(avgAng>45&&avgAng<90)countFnFrame(2);
                if(avgAng<-45&&avgAng>-90)countFnFrame(3);
                stopTrackPalm();
            }

            if (real_finger_count==3) {
                float angel_temp = getAvgFingersAngleByIndex(1,3,outputHandPos);
                if (angel_temp > -30 && angel_temp < 30) countFnFrame(1);
            }

            if (real_finger_count==4) {
                float angel_temp = getAvgFingersAngleByIndex(1,4,outputHandPos);
                if (angel_temp > -30 && angel_temp < 30) countFnFrame(0);
            }

            if (real_finger_count==5) {
                float avgAngle = getAvgFingersAngle(outputHandPos);
                if (avgAngle > -10 && avgAngle < 20) countFnFrame(8); // for standby of post 0,1,2,3 don't remove this line!

                if(avgAngle > 30) {
                    countFnFrame(4);
                } else if(avgAngle > 45) {
                    countFnFrame(4);
                    countFnFrame(4);
                } else if(avgAngle < -30) {
                    countFnFrame(5);
                } else if(avgAngle < -45) {
                    countFnFrame(5);
                    countFnFrame(5);
                }

                //for track hand position
                if(!isPalmPosMarked) startTrackPalm();

                Point nowPalmPos = (bigOutputSkeletonPos.first+bigOutputPos);
                float palmLineAngle = getFingerAngle(make_pair(nowPalmPos, palmPos));

                if(isInTaskView()){
                    if(cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && palmLineAngle > -30 && palmLineAngle < 30){
                        // palm up
                        press(13);
                        stopTrackPalm();
                        cooldown = 60;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && (palmLineAngle < -150 || palmLineAngle > 150)){
                        // palm down

                        stopTrackPalm();
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle <= -60 && palmLineAngle >= -120 ){
                        // palm left
                        press(37);
                        if(getTrackedPalmDistance() > 80) cooldown = 5;
                        else cooldown = 10;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle >= 60 && palmLineAngle <=120){
                        // palm right
                        press(39);
                        if(getTrackedPalmDistance() > 80) cooldown = 5;
                        else cooldown = 10;
                    } else if (avgAngle > -10 && avgAngle < 20){
                        countFnFrame(8); // for standby of post 0,1,2,3 don't remove this line!
                    }
                } else if(isInPowerPoint()){
                    if(cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && palmLineAngle > -30 && palmLineAngle < 30){
                        // palm up
                        press(116);
                        stopTrackPalm();
                        cooldown = 60;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && (palmLineAngle < -150 || palmLineAngle > 150)){
                        // palm down
                        press(27);
                        stopTrackPalm();
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle <= -60 && palmLineAngle >= -120 ){
                        // palm left
                        press(37);
                        if(getTrackedPalmDistance() > 80) cooldown = 10;
                        else cooldown = 20;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle >= 60 && palmLineAngle <=120){
                        // palm right
                        enterTaskView();
                        cout << "TaskView" << endl;
                    } else if (avgAngle > -10 && avgAngle < 20){
                        countFnFrame(8); // for standby of post 0,1,2,3 don't remove this line!
                    }
                } else if(isInSlideShow()){
                    if(cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && palmLineAngle > -30 && palmLineAngle < 30){
                        // palm up
                        press(37);
                        stopTrackPalm();
                        cooldown = 60;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && (palmLineAngle < -150 || palmLineAngle > 150)){
                        // palm down

                        stopTrackPalm();
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle <= -60 && palmLineAngle >= -120 ){
                        // palm left
                        press(37);
                        if(getTrackedPalmDistance() > 80) cooldown = 5;
                        else cooldown = 10;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 50 && palmLineAngle >= 60 && palmLineAngle <=120){
                        // palm right
                        press(39);
                        if(getTrackedPalmDistance() > 80) cooldown = 5;
                        else cooldown = 10;
                    } else if (avgAngle > -10 && avgAngle < 20){
                        countFnFrame(8); // for standby of post 0,1,2,3 don't remove this line!
                    }
                } else {
                    if(cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && palmLineAngle > -30 && palmLineAngle < 30){
                        // palm up
                        triggerFunction(6);
                        stopTrackPalm();
                        cooldown = 60;
                    } else if (cooldown<=0 && getTrackedPalmDistance() > 80 && avgAngle > -30 && avgAngle < 30 && (palmLineAngle < -150 || palmLineAngle > 150)){
                        // palm down
                        triggerFunction(7);
                        stopTrackPalm();
                        cooldown = 60;
                    } else if (getTrackedPalmDistance() > 80 && palmLineAngle <= -60 && palmLineAngle >= -120 ){
                        // palm left
                        cout << "no ";
                        stopTrackPalm();
                    } else if (getTrackedPalmDistance() > 80 && palmLineAngle >= 60 && palmLineAngle <=120){
                        // palm right
                        enterTaskView();
                        cout << "TaskView" << endl;
                        stopTrackPalm();
                    } else if (avgAngle > -10 && avgAngle < 20){
                        countFnFrame(8); // for standby of post 0,1,2,3 don't remove this line!
                    }
                }
            }

            if(cooldown>0) cooldown--;

            showBuffer(shownCaptureFrame);
            showMode(shownCaptureFrame);
            showAngel(shownCaptureFrame);
            showPalmTrack(shownCaptureFrame);

            imshow("Webcam", shownCaptureFrame);

            Mat output = Mat(captureFrame.rows, captureFrame.cols, CV_8UC3);
            for(int i=0; i<output.rows; i++) {
                for(int j=0; j<output.cols; j++) {
                    bool found = false;
                    Vec3b c = captureFrameHSV.at<Vec3b>(i,j);
                    for(int cc=0; cc<maxb.size(); cc++) {
                        if(c[0] >= minb[cc][0] && c[1] >= minb[cc][1] && c[2] >= minb[cc][2] &&
                                c[0] <= maxb[cc][0] && c[1] <= maxb[cc][1] && c[2] <= maxb[cc][2]) {
                            found = true;
                            break;
                        }
                    }
                    if(found) {
                        output.at<Vec3b>(i,j) = Vec3b(255,255,255);
                    } else {
                        output.at<Vec3b>(i,j) = Vec3b(0,0,0);
                    }
                }
            }
            Mat output2 = output.clone();
//            output = output2.clone();
//            myErode(output2,output,1);
//            output2 = output.clone();
//            myErode(output, output2,1);
//            output = output2.clone();
//            myDilate(output2,output,1);
//            bg1 = getNonBackground(captureFrameHSV);
//            bg2 = bg1.clone();
//            myErode(bg1,bg2,4);
//            bg1 = bg2.clone();
//            myErode(bg2,bg1,4);
//            bg2 = bg1.clone();
//            myDilate(bg1,bg2,8);
//            bg1 = bg2.clone();
//            myDilate(bg2,bg1,8);
            bg1 = Mat(output2.rows,output2.cols,CV_8UC3,Vec3b(255,255,255));
            bounds = myConnectedComponents(output2, output, bg1);
            Mat bigOutput;
            bigOutputPos = getBiggestComponent(output,bounds,bigOutput);
            for(int i=0; i<bg1.rows; i++) {
                for(int j=0; j<bg1.cols; j++) {
                    if(bg1.at<Vec3b>(i,j)[0] > 128 && (2*i+j)%4 == 0) {
                        output.at<Vec3b>(i,j) = Vec3b(70,70,70);
                    }
                }
            }
            fillHole(bigOutput,bigOutput);
            Mat bigOutputGrad;
            bigOutputSkeletonPos = biggestCenter(bigOutput,bigOutputGrad,bigOutputSkeleton);
            outputHandPos = getHand(bigOutputSkeleton,bigOutputSkeletonPos);
            //myDilate(bigOutputSkeleton.clone(),bigOutputSkeleton,1);
            resize(output,output,Size(640,480));
            //resize(bigOutput,bigOutput,Size(bigOutput.cols*2,bigOutput.rows*2));
            resize(bigOutputSkeleton,bigOutputSkeleton,Size(bigOutputSkeleton.cols*2,bigOutputSkeleton.rows*2));
            imshow("Output", bigOutputSkeleton);
            if(waitKey(5) == 27) {
                exit(0);
                break;
            }
            if(waitKey(5) == 13) {
                maxb.clear();
                minb.clear();
                fnFrameCountBuffer.clear();
            }
        }
    }
    return 0;
}
