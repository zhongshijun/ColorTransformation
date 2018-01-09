#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

typedef bool BOOL;
typedef long LONG;
typedef unsigned char BYTE;
typedef BYTE* LPBYTE;


using namespace std;
using namespace cv;


void RgbToLab(BYTE R,BYTE G,BYTE B,double& l,double& a,double& b)
{
    double L = 0.3811*R + 0.5783*G + 0.0402*B;
    double M = 0.1967*R + 0.7244*G + 0.0782*B;
    double S = 0.0241*R + 0.1288*G + 0.8444*B;

    //若RGB值均为0，则LMS为0，防止数学错误log0
    if(L!=0) L = log10(L);
    if(M!=0) M = log10(M);
    if(S!=0) S = log10(S);

    l = (L + M + S)/sqrt(3);
    a = (L + M - 2*S)/sqrt(6);
    b = (L - M)/sqrt(2);

}

void LabToRgb(double l,double a,double b,BYTE& R,BYTE& G,BYTE& B)
{
    l /= sqrt(3);
    a /= sqrt(6);
    b /= sqrt(2);
    double L = l + a + b;
    double M = l + a - b;
    double S = l - 2*a;

    L = pow(10,L);
    M = pow(10,M);
    S = pow(10,S);

    double dR = 4.4679*L - 3.5873*M + 0.1193*S;
    double dG = -1.2186*L + 2.3809*M - 0.1624*S;
    double dB = 0.0497*L - 0.2439*M + 1.2045*S;

    //防止溢出，若求得RGB值大于255则置为255，若小于0则置为0
    if (dR>255) R=255;
    else if (dR<0) R=0;
    else R = BYTE(dR);

    if (dG>255) G=255;
    else if (dG<0) G=0;
    else G = BYTE(dG);

    if (dB>255) B=255;
    else if (dB<0) B=0;
    else B = BYTE(dB);
}

BYTE RgbToGray(BYTE R,BYTE G,BYTE B)
{
    int Gray = int(0.29900*R + 0.58700*G + 0.11400*B +0.5);
    if (Gray>255) Gray=255;
    if (Gray<0) Gray=0;
    return (BYTE)Gray;
}

//全局平均值及标准差
void AverageVariance(double* lpLab,int Width,int Height,double& al,double& aa,double& ab,double& vl,double& va,double& vb)
{
    double suml=0;
    double suma=0;
    double sumb=0;
    double lsuml=0;
    double lsuma=0;
    double lsumb=0;

    //分行求平均，避免和过大而溢出
    for(int j=0;j<Height;j++)
    {
        for(int i=0;i<Width;i++)
        {
            lsuml+=lpLab[(j*Width+i)*3];
            lsuma+=lpLab[(j*Width+i)*3+1];
            lsumb+=lpLab[(j*Width+i)*3+2];
        }
        suml += lsuml/Width;
        suma += lsuma/Width;
        sumb += lsumb/Width;
        lsuml=lsuma=lsumb=0;
    }
    al = suml/Height;
    aa = suma/Height;
    ab = sumb/Height;

    suml=suma=sumb=0;
    for(int i=0;i<Width*Height;i++)
    {
        suml += pow(lpLab[i*3]-al,2);
        suma += pow(lpLab[i*3+1]-aa,2);
        sumb += pow(lpLab[i*3+2]-ab,2);
    }
    vl = sqrt(suml);
    va = sqrt(suma);
    vb = sqrt(sumb);
}

//分类求平均及标准差
void AverageVariance(LPBYTE lpDIBBits,LONG lmageWidth,LONG lmageHeight,LPBYTE belong,double* al,double* aa,double* ab,double* vl,double* va,double* vb,int classnum)
{
    int i,j,k,nindex;
    double l,a,b;
    double* suml=new double[classnum];
    double* suma=new double[classnum];
    double* sumb=new double[classnum];
    double* num=new double[classnum];
    for(k=0;k<classnum;k++)
    {
        suml[k]=0;suma[k]=0;sumb[k]=0;num[k]=0;
        al[k]=0;aa[k]=0;ab[k]=0;vl[k]=0;va[k]=0;vb[k]=0;
    }

    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],l,a,b);
            suml[belong[nindex]]+=l;
            suma[belong[nindex]]+=a;
            sumb[belong[nindex]]+=b;
            num[belong[nindex]]++;
        }
    }
    for(k=0;k<classnum;k++)
    {
        al[k]=suml[k]/num[k];
        aa[k]=suma[k]/num[k];
        ab[k]=sumb[k]/num[k];
        suml[k]=0;suma[k]=0;sumb[k]=0;
    }

    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],l,a,b);
            suml[belong[nindex]]+=pow(l-al[belong[nindex]],2);
            suma[belong[nindex]]+=pow(a-aa[belong[nindex]],2);
            sumb[belong[nindex]]+=pow(b-ab[belong[nindex]],2);
        }
    }
    for(k=0;k<classnum;k++)
    {
        vl[k]=sqrt(suml[k]);
        va[k]=sqrt(suma[k]);
        vb[k]=sqrt(sumb[k]);
    }
}

double DistanceLab(double l1,double a1,double b1,double l2,double a2,double b2)
{
    double lx=l1-l2;
    double ax=a1-a2;
    double bx=b1-b2;
    if (lx<0) lx=-lx;
    if (ax<0) ax=-ax;
    if (bx<0) bx=-bx;
    return lx+ax+bx;
}

BOOL TranRein(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2, LONG lmageWidth2, LONG lmageHeight2,LPBYTE lpDIBBits3)
{
    int i;
    int j;
    int nindex;
    double al,aa,ab,vl,va,vb,al2,aa2,ab2,vl2,va2,vb2;
    double* lpImageLab = new  double[lmageWidth*lmageHeight*3];
    double* lpImageLab2 = new  double[lmageWidth2*lmageHeight2*3];
    double* lpImageLab3 = new  double[lmageWidth*lmageHeight*3];

    //目标图像转换为lab，并求lab的均值及标准差
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],
                lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2]);
        }
    }
    AverageVariance(lpImageLab,lmageWidth,lmageHeight,al,aa,ab,vl,va,vb);

    //源图像转换为lab，并求lab的均值及标准差
    for(j = 0;j <lmageHeight2; j++)
    {
        for(i = 0; i <lmageWidth2; i++)
        {
            nindex=((lmageHeight2-j-1)*lmageWidth2+i);
            RgbToLab(lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0],
                lpImageLab2[nindex*3+0],lpImageLab2[nindex*3+1],lpImageLab2[nindex*3+2]);
        }
    }
    AverageVariance(lpImageLab2,lmageWidth2,lmageHeight2,al2,aa2,ab2,vl2,va2,vb2);

    //求结果图像的lab
    for(i = 0;i <lmageWidth*lmageHeight; i++)
    {
        lpImageLab3[i*3+0] = (lpImageLab[i*3+0] - al) * vl2/vl + al2;
        lpImageLab3[i*3+1] = (lpImageLab[i*3+1] - aa) * va2/va + aa2;
        lpImageLab3[i*3+2] = (lpImageLab[i*3+2] - ab) * vb2/vb + ab2;
    }

    //将结果图像的lab转换为RGB
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);

            LabToRgb(lpImageLab3[nindex*3+0],lpImageLab3[nindex*3+1],lpImageLab3[nindex*3+2],
                lpDIBBits3[nindex*3+2],lpDIBBits3[nindex*3+1],lpDIBBits3[nindex*3+0]);
        }
    }
    return true;
}

BOOL HistogramSpecification(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2, LONG lmageWidth2, LONG lmageHeight2,LPBYTE lpDIBBits3)
{
    int i,j,k,l;
    int sumGray=0;
    int sumGray2=0;
    int bGrayMap[256]={0};
    long histGray[256]={0};
    long histGray2[256]={0};
    //统计直方图
    for(j=0;j<lmageHeight;j++)
    {
        for(i=0;i<lmageWidth;i++)
        {
            int nindex=((lmageHeight-j-1)*lmageWidth+i);
            histGray[RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0])]++;
        }
    }
    for(j=0;j<lmageHeight2;j++)
    {
        for(i=0;i<lmageWidth2;i++)
        {
            int nindex=((lmageHeight2-j-1)*lmageWidth2+i);
            histGray2[RgbToGray(lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0])]++;
        }
    }

    //灰度映射
    for(i=1;i<256;i++)
    {
        sumGray=0;
        for(j=0;j<=i;j++)
        {
            sumGray+=histGray[j];
        }
//		for(k=bGrayMap[i-1];k<256;k++)
        for(k=1;k<256;k++)
        {
            sumGray2=0;
            for(l=0;l<=k;l++)
            {
                sumGray2+=histGray2[l];
            }
            if(((double)sumGray2/(lmageWidth2*lmageHeight2))>=((double)sumGray/(lmageWidth*lmageHeight))) break;
        }
        bGrayMap[i]=k;
    }

    //重新定义灰度
    for(j=0;j<lmageHeight;j++)
    {
        for(i=0;i<lmageWidth;i++)
        {
            int nindex=((lmageHeight-j-1)*lmageWidth+i);
            lpDIBBits3[nindex*3+2] = lpDIBBits3[nindex*3+1] = lpDIBBits3[nindex*3+0] = bGrayMap[RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0])];
        }
    }
    return true;
}

BOOL GrayMapping(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2, LONG lmageWidth2, LONG lmageHeight2,LPBYTE lpDIBBits3)
{
    int i;
    int j;
    int nindex;
    int num[256]={0};
    double l[256]={0};
    double a[256]={0};
    double b[256]={0};
    //
    for(j = 0;j <lmageHeight2; j++)
    {
        for(i = 0; i <lmageWidth2; i++)
        {
            nindex=((lmageHeight2-j-1)*lmageWidth2+i);
            BYTE gray=RgbToGray(lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0]);
            num[gray]++;
            double ll,aa,bb;
            RgbToLab(lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0],ll,aa,bb);
            l[gray]+=ll;
            a[gray]+=aa;
            b[gray]+=bb;
        }
    }
    for(i=0;i<256;i++)
    {
        l[i]/=num[i];
        a[i]/=num[i];
        b[i]/=num[i];
    }

    //
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            BYTE gray=RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0]);
//			double ll,aa,bb;
//			RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],ll,aa,bb);
            if((l[gray]!=0)||(a[gray]!=0)||(b[gray]!=0))
            LabToRgb(l[gray],a[gray],b[gray],lpDIBBits3[nindex*3+2],lpDIBBits3[nindex*3+1],lpDIBBits3[nindex*3+0]);
        }
    }
    return true;
}

BOOL GrayDIB(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2)
{
    int i,j,nindex;
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            lpDIBBits2[nindex*3+2]=lpDIBBits2[nindex*3+1]=lpDIBBits2[nindex*3+0]=
                RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0]);
        }
    }
    return true;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL KMeansCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE belong,LPBYTE center,int classnum)
{
    int i,j,nindex;
    int k=0;
    int LOOP=1000;
    long x,y;
    LPBYTE center2=new BYTE[classnum];
    long* num=new long[classnum];
    long int* sum=new long int[classnum];
    for(i=0;i<classnum;i++)
    {
        center[i]=0;
        center2[i]=0;
        num[i]=0;
        sum[i]=0;
    }
//    srand(time(NULL));

    //初始化聚类中心
    for(i=0;i<classnum;i++)
    {
        x=rand()%lmageWidth;
        y=rand()%lmageHeight;
        nindex=((lmageHeight-y-1)*lmageWidth+x);
        BYTE gray=RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0]);
        center[i]=gray;
        for(j=0;j<i;j++)
            if(gray==center[j]) {i--;break;}
    }

    //重新聚类、计算聚类中心、直至聚类中心不再变化
    while(k!=classnum && LOOP--)
    {
        for(j = 0;j <lmageHeight; j++)
        {
            for(i = 0; i <lmageWidth; i++)
            {
                nindex=((lmageHeight-j-1)*lmageWidth+i);
                BYTE gray=RgbToGray(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0]);
                //means_Assign();
                BYTE distance=255;
                int blg=-1;
                for(k=0;k<classnum;k++)
                {
                    if (distance>abs(gray-center[k]))  {distance=abs(gray-center[k]);blg=k;}
                }
                belong[nindex]=blg;
                num[blg]++;
                sum[blg]+=gray;
            }
        }


        //means_Center();
        for(i=0;i<classnum;i++)
        {
            center2[i]=BYTE(sum[i]/num[i]);
        }
        for(k=0;k<classnum;k++)
        {
            if(center[k]!=center2[k]) break;
        }
        for(i=0;i<classnum;i++)
        {
            center[i]=center2[i];
        }
    }


    return true;
}

BOOL KMeansCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2)
{
    int classnum=2;//初始化聚类个数
    int i,j,nindex;
    LPBYTE belong=new BYTE[lmageWidth*lmageHeight];
    LPBYTE center=new BYTE[classnum];
    KMeansCluster(lpDIBBits,lmageWidth,lmageHeight,belong,center,classnum);
    //对聚类结果图像赋值

    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            lpDIBBits2[nindex*3+2]=lpDIBBits2[nindex*3+1]=lpDIBBits2[nindex*3+0]=center[belong[nindex]];
        }
    }
    return true;

}


////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL HCMCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE belong,double* center,int classnum)
{
    int i,j,nindex;
    int k=0;
    int LOOP=500;
    double* center2=new double[classnum*3];
    long x,y;
    long* num=new long[classnum];
    double* sum=new double[classnum*3];
    double* lpImageLab = new  double[lmageWidth*lmageHeight*3];
    for(i=0;i<classnum;i++)
    {
        center[i*3+0]=0;
        center[i*3+1]=0;
        center[i*3+2]=0;
        center2[i*3+0]=0;
        center2[i*3+1]=0;
        center2[i*3+2]=0;
        num[i]=0;
        sum[i*3+0]=0;
        sum[i*3+1]=0;
        sum[i*3+2]=0;
    }
//    srand(time(NULL));

    //初始化聚类中心
    for(i=0;i<classnum;i++)
    {
        x=rand()%lmageWidth;
        y=rand()%lmageHeight;
        nindex=((lmageHeight-y-1)*lmageWidth+x);
        RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],center[i*3+0],center[i*3+1],center[i*3+2]);
        for(j=0;j<i;j++)
        {
            double dis=DistanceLab(center[i*3+0],center[i*3+1],center[i*3+2],center[j*3+0],center[j*3+1],center[j*3+2]);
            if(dis<0.2) {i--;break;}//限值公式 暂取限值为1待优化
        }
    }

    //重新聚类、计算聚类中心、直至聚类中心距离小于限值e暂定0.1待优化
    while(k!=classnum && LOOP--)//限值公式
    {
        for(j = 0;j <lmageHeight; j++)
        {
            for(i = 0; i <lmageWidth; i++)
            {
                nindex=((lmageHeight-j-1)*lmageWidth+i);
                RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],
                    lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2]);
                //means_Assign();
                double distance=3333333;//取某一较大的距离，暂取3333333，
                int blg=-1;//隶属度
                for(k=0;k<classnum;k++)
                {
                    double dis=DistanceLab(lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2],
                        center[k*3+0],center[k*3+1],center[k*3+2]);
                    if (distance>dis)  {distance=dis;blg=k;}
                }
                belong[nindex]=blg;
                num[blg]++;
                sum[blg*3+0]+=lpImageLab[nindex*3+0];
                sum[blg*3+1]+=lpImageLab[nindex*3+1];
                sum[blg*3+2]+=lpImageLab[nindex*3+2];
            }
        }


        //means_Center();
        for(i=0;i<classnum;i++)
        {
            center2[i*3+0]=sum[i*3+0]/num[i];
            center2[i*3+1]=sum[i*3+1]/num[i];
            center2[i*3+2]=sum[i*3+2]/num[i];
        }
        //判断循环终止条件
        for(k=0;k<classnum;k++)
        {
            if(DistanceLab(center[k*3+0],center[k*3+1],center[k*3+2],center2[k*3+0],center2[k*3+1],center2[k*3+2])>0.1) break;//限值e暂定0.1待优化
        }
        for(i=0;i<classnum*3;i++)
        {
            center[i]=center2[i];
        }
    }
    return true;
}

BOOL HCMCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2)
{
    int classnum=2;
    int i,j,nindex;
    LPBYTE belong=new BYTE[lmageWidth*lmageHeight];//隶属矩阵
    double* center=new double[classnum*3];
    HCMCluster(lpDIBBits,lmageWidth,lmageHeight,belong,center,classnum);
    //对聚类结果图像赋值
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            LabToRgb(center[belong[nindex]*3+0],center[belong[nindex]*3+1],center[belong[nindex]*3+2],
                lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0]);
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////
BOOL FCMCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,double* belong,double* center,int classnum,int m)
{
    int i,j,l,nindex;//循环控制变量
    int k=0;
    int LOOP=500;
    double* center2=new double[classnum*3];//聚类中心
    long x,y;//随机确定聚类中心坐标
    long* num=new long[classnum];//每个类的像素个数
    double* lpImageLab = new  double[lmageWidth*lmageHeight*3];
    double sumu,suml,suma,sumb;

    //初始化聚类中心
    for(i=0;i<classnum;i++)
    {
        x=rand()%lmageWidth;
        y=rand()%lmageHeight;
        nindex=((lmageHeight-y-1)*lmageWidth+x);
        RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],center[i*3+0],center[i*3+1],center[i*3+2]);
        for(j=0;j<i;j++)
        {
            double dis=DistanceLab(center[i*3+0],center[i*3+1],center[i*3+2],center[j*3+0],center[j*3+1],center[j*3+2]);
            if(dis<0.2) {i--;break;}//限值公式 暂取限值为1待优化 对初始化聚类中心的选择非常关键
        }
    }

    //计算隶属度矩阵、更新聚类中心、直至前后聚类中心距离小于限值e暂定0.1待优化
    while(k!=classnum && LOOP--)//限值公式
    {
        //计算隶属度矩阵
        for(j = 0;j <lmageHeight; j++)
        {
            for(i = 0; i <lmageWidth; i++)
            {
                nindex=((lmageHeight-j-1)*lmageWidth+i);
                RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],
                    lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2]);
                //means_Assign();
                double blg=-1;//隶属度
                for(k=0;k<classnum;k++)
                {
                    sumu=0;
                    double dis1=DistanceLab(lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2],center[k*3+0],center[k*3+1],center[k*3+2]);
                    if (dis1==0) {belong[lmageWidth*lmageHeight*k+nindex]=1;continue;}
                    for(l=0;l<classnum;l++)
                    {
                        double dis2=DistanceLab(lpImageLab[nindex*3+0],lpImageLab[nindex*3+1],lpImageLab[nindex*3+2],center[l*3+0],center[l*3+1],center[l*3+2]);
                        if (dis2==0) break;
                        sumu+=pow((dis1*dis1)/(dis2*dis2),1.0/(m-1));
                    }
                    if (l!=classnum) {belong[lmageWidth*lmageHeight*k+nindex]=0;continue;}
                    belong[lmageWidth*lmageHeight*k+nindex]=1/sumu;
                }
            }
        }

        //更新聚类中心
        for(k=0;k<classnum;k++)
        {
            suml=suma=sumb=sumu=0;
            for(j = 0;j <lmageHeight; j++)
            {
                for(i = 0; i <lmageWidth; i++)
                {
                    nindex=((lmageHeight-j-1)*lmageWidth+i);
                    suml+=pow(belong[lmageWidth*lmageHeight*k+nindex],m)*lpImageLab[nindex*3+0];
                    suma+=pow(belong[lmageWidth*lmageHeight*k+nindex],m)*lpImageLab[nindex*3+1];
                    sumb+=pow(belong[lmageWidth*lmageHeight*k+nindex],m)*lpImageLab[nindex*3+2];
                    sumu+=pow(belong[lmageWidth*lmageHeight*k+nindex],m);
                }
            }
            center2[k*3+0]=suml/sumu;
            center2[k*3+1]=suma/sumu;
            center2[k*3+2]=sumb/sumu;
        }

        //判断循环终止条件
        for(k=0;k<classnum;k++)
        {
            if(DistanceLab(center[k*3+0],center[k*3+1],center[k*3+2],center2[k*3+0],center2[k*3+1],center2[k*3+2])>0.1) break;//限值e暂定0.1待优化
        }
        for(i=0;i<classnum*3;i++)
        {
            center[i]=center2[i];
        }

    }

    return true;
}


BOOL FCMCluster(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2)
{
    int classnum=2;//初始化聚类个数
    int m=2;//加权指数
    int i,j,k,nindex;//循环控制变量
    double* belong=new double[lmageWidth*lmageHeight*classnum];//隶属度矩阵
    double* center=new double[classnum*3];
    double suml,suma,sumb;
    FCMCluster(lpDIBBits,lmageWidth,lmageHeight,belong,center,classnum,m);
    //根据聚类中心和隶属度矩阵生成聚类结果图像
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            suml=suma=sumb=0;
            for(k=0;k<classnum;k++)
            {
                suml+=belong[lmageWidth*lmageHeight*k+nindex]*center[k*3+0];
                suma+=belong[lmageWidth*lmageHeight*k+nindex]*center[k*3+1];
                sumb+=belong[lmageWidth*lmageHeight*k+nindex]*center[k*3+2];
            }

            LabToRgb(suml,suma,sumb,lpDIBBits2[nindex*3+2],lpDIBBits2[nindex*3+1],lpDIBBits2[nindex*3+0]);
        }
    }
    return true;
}

BOOL TranKMeans(LPBYTE lpDIBBits, LONG lmageWidth, LONG lmageHeight,LPBYTE lpDIBBits2, LONG lmageWidth2, LONG lmageHeight2,LPBYTE lpDIBBits3)
{


    int classnum=2;
    int i,j,nindex;
    double l,a,b;
    LPBYTE belong=new BYTE[lmageWidth*lmageHeight];
    LPBYTE belong2=new BYTE[lmageWidth2*lmageHeight2];
    LPBYTE center=new BYTE[classnum];
    LPBYTE center2=new BYTE[classnum];
    int* clustermap=new int[classnum];
    KMeansCluster(lpDIBBits,lmageWidth,lmageHeight,belong,center,classnum);
    KMeansCluster(lpDIBBits2,lmageWidth2,lmageHeight2,belong2,center2,classnum);


    //ClusterMap(center,center2);
    for(i=0;i<classnum;i++)
    {
        BYTE distance=255;
        int map=-1;
        for(j=0;j<classnum;j++)
        {
            if (distance>abs(center[i]-center2[j])) {distance=abs(center[i]-center2[j]);map=j;}
        }
        clustermap[i]=map;
    }




    //TranColor(belong,belong2,center,center2);
    double* al=new double[classnum];
    double* aa=new double[classnum];
    double* ab=new double[classnum];
    double* vl=new double[classnum];
    double* va=new double[classnum];
    double* vb=new double[classnum];
    double* al2=new double[classnum];
    double* aa2=new double[classnum];
    double* ab2=new double[classnum];
    double* vl2=new double[classnum];
    double* va2=new double[classnum];
    double* vb2=new double[classnum];
    AverageVariance(lpDIBBits,lmageWidth,lmageHeight,belong,al,aa,ab,vl,va,vb,classnum);
    AverageVariance(lpDIBBits2,lmageWidth2,lmageHeight2,belong2,al2,aa2,ab2,vl2,va2,vb2,classnum);


    //求结果图像的lab
    for(j = 0;j <lmageHeight; j++)
    {
        for(i = 0; i <lmageWidth; i++)
        {
            nindex=((lmageHeight-j-1)*lmageWidth+i);
            RgbToLab(lpDIBBits[nindex*3+2],lpDIBBits[nindex*3+1],lpDIBBits[nindex*3+0],l,a,b);
//			l = (l - al[belong[nindex]]) * vl2[clustermap[belong[nindex]]]/vl[belong[nindex]] + al2[clustermap[belong[nindex]]];
            a = (a - aa[belong[nindex]]) * va2[clustermap[belong[nindex]]]/va[belong[nindex]] + aa2[clustermap[belong[nindex]]];
            b = (b - ab[belong[nindex]]) * vb2[clustermap[belong[nindex]]]/vb[belong[nindex]] +ab2[clustermap[belong[nindex]]];
            LabToRgb(l,a,b,lpDIBBits3[nindex*3+2],lpDIBBits3[nindex*3+1],lpDIBBits3[nindex*3+0]);
        }
    }

//    cout << endl << endl;
//    for (int i = 0;i < lmageHeight * lmageWidth; ++i) {
//        cout << float(lpDIBBits3[i]) << " ";
//        if (i % 25 == 0) cout << endl;
//    }

    return true;
}





int main()
{

    Mat srcImg = imread("F:\\test.jpg");
    Mat dstImg = imread("F:\\dest.jpg");
//    resize(srcImg, srcImg, Size(25, 25));
    Mat result(srcImg.size(), srcImg.type());

//    cout << "rows: " << result.rows << "  cols: " << result.cols << endl;
//    for (int i = 0;i < result.rows; ++i) {
//        for (int j = 0;j < result.cols; ++j) {
//            cout << float(result.data[i*result.rows + j]) << " ";
//        }
//        cout << endl;
//    }

    TranKMeans(srcImg.data, srcImg.cols, srcImg.rows, dstImg.data, dstImg.cols, dstImg.rows, result.data);
    imshow("srcImg", srcImg);
    imshow("dest", dstImg);
    imshow("test", result);
    waitKey(0);

    //    LPBYTE lpbyte = new BYTE[srcImg.rows * srcImg.cols];

//    for (int i = 0;i < srcImg.rows; ++i) {
//        for (int j = 0;j < srcImg.cols;++j) {
//            int nindex = i * srcImg.rows + 3 * j;
//            lpbyte[nindex] = srcImg.at<cv::Vec3b>(i,j)[0];
//            lpbyte[nindex + 1] = srcImg.at<cv::Vec3b>(i,j)[1];
//            lpbyte[nindex + 2] = srcImg.at<cv::Vec3b>(i,j)[2];
//        }
//    }

//    for (int i = 0;i < srcImg.rows; ++i) {
//        cout << float(srcImg.data[i]) << " ";
//        //        cout << lpbyte[i] << " ";
////        if (i % 10) cout << endl;

//    }
//    cout<< "finish!" << endl;
////    imshow("test", srcImg);
////    waitKey(0);


    return 0;
}
