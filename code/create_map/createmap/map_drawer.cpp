#include "map_drawer.h"


map_drawer::map_drawer()
{

}

map_drawer::map_drawer(Mat img_in,string map_in,string imgsavepath)
{
    img_in_=img_in.clone();
    img_with_map_=img_in_.clone();
    height_=img_in.rows;
    width_=img_in.cols;
    map_=MatrixXf::Zero(height_,width_);
    pointvec.clear();
    map_in_=map_in;
    img_save_path_=imgsavepath;
}


void map_drawer::update_img_with_map()
{
    ifstream inf;
    inf.open(map_in_);
    string line;
    img_with_map_=img_in_.clone();
    while(getline(inf,line))
    {
        int x,y,w,h;
        char a,b,c,d,e;
        float p,q;
        strstream ss;
        ss<<line;
        ss>>x>>a>>y>>b>>w>>c>>h>>d>>p>>e>>q;
        cout<<x<<" "<<y<<" "<<w<<" "<<h<<" "<<p<<" "<<q<<" "<<endl;
        for (int ww=0;ww<width_;ww++)
        {
            for(int hh=0;hh<height_;hh++)
            {
                float dis=0;
                float dis1sq=(ww-x)*(ww-x)+(hh-y)*(hh-y);
                float dis2sq=(ww-w)*(ww-w)+(hh-h)*(hh-h);
                float dis_bet=(w-x)*(w-x)+(h-y)*(h-y);
                if(dis2sq+dis_bet<dis1sq)
                {
                    dis=pow(dis2sq,0.5);
                }
                else if(dis1sq+dis_bet<dis2sq)
                {
                    dis=pow(dis1sq,0.5);
                }
                else
                {
                    float ba=pow(dis1sq,0.5);
                    float bb=pow(dis2sq,0.5);
                    float bc=pow(dis_bet,0.5);
                    float cos=(ba*ba+bc*bc-bb*bb)/(2.0*ba*bc);
                    float sin=pow(1-cos*cos,0.5);
                    dis=ba*sin;
                }
                if(p*exp(-q*dis*dis)>map_(hh,ww))
                {
                    map_(hh,ww)=p*exp(-q*dis*dis);
                }

            }
        }
    }

    string mapout=map_in_.substr(0,map_in_.length()-3)+"map";
    //cout<<mapout<<endl;
    ofstream ouf;
    ouf.open(mapout);
    for(int hh=0;hh<height_;hh++)
    {
        for(int ww=0;ww<width_;ww++)
        {
            ouf<<map_(hh,ww)<<" ";
            Vector3i color_v=color(map_(hh,ww));
            img_with_map_.at<Vec3b>(hh,ww)[0]/=2.0;
            img_with_map_.at<Vec3b>(hh,ww)[1]/=2.0;
            img_with_map_.at<Vec3b>(hh,ww)[2]/=2.0;
            img_with_map_.at<Vec3b>(hh,ww)[0]+=color_v(0)*0.5;
            img_with_map_.at<Vec3b>(hh,ww)[1]+=color_v(1)*0.5;
            img_with_map_.at<Vec3b>(hh,ww)[2]+=color_v(2)*0.5;
        }
        ouf<<endl;
    }
    ouf.close();
    imshow("yousee",img_with_map_);
    waitKey();
    imwrite(img_save_path_,img_with_map_);
}

void map_drawer::update_the_map(Vector2i p1,Vector2i p2)
{
    pointvec.push_back(p1);
    pointvec.push_back(p2);
}


void map_drawer::contrust_the_map()
{
    int x,y,w,h;
    x=width_/3;
    w=2*width_/3;
    y=h=2*height_/3;
    for(int i=y-2;i<=y+2;i++)
    {
        for(int j=x-2;j<=x+2;j++)
        {
            img_with_map_.at<Vec3b>(i,j)[0]=255;
            img_with_map_.at<Vec3b>(i,j)[1]=0;
            img_with_map_.at<Vec3b>(i,j)[2]=0;
        }
    }
    for(int i=h-2;i<=h+2;i++)
    {
        for(int j=w-2;j<=w+2;j++)
        {
            img_with_map_.at<Vec3b>(i,j)[0]=0;
            img_with_map_.at<Vec3b>(i,j)[1]=255;
            img_with_map_.at<Vec3b>(i,j)[2]=0;
        }
    }
    imshow("you see",img_in_);
    char c=waitKey();
    while(c=='w' || c=='a' || c=='s' || c=='d'
          ||  c=='j' || c=='k' || c=='l' || c=='i' || c=='b')
    {
        imshow("you see",img_with_map_);
        if(c=='w' && y>2)
        {

            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            y--;
            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=255;
                    img_with_map_.at<Vec3b>(i,j)[1]=0;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='s' && y<height_-3)
        {

            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            y++;
            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=255;
                    img_with_map_.at<Vec3b>(i,j)[1]=0;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='a' && x>2)
        {

            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            x--;
            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=255;
                    img_with_map_.at<Vec3b>(i,j)[1]=0;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='d' && x<width_-3)
        {

            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            x++;
            for(int i=y-2;i<=y+2;i++)
            {
                for(int j=x-2;j<=x+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=255;
                    img_with_map_.at<Vec3b>(i,j)[1]=0;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='i' && h>2)
        {

            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            h--;
            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=0;
                    img_with_map_.at<Vec3b>(i,j)[1]=255;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='k' && h<height_-3)
        {

            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            h++;
            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=0;
                    img_with_map_.at<Vec3b>(i,j)[1]=255;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='j' && w>2)
        {

            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            w--;
            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=0;
                    img_with_map_.at<Vec3b>(i,j)[1]=255;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='l' && w<width_-3)
        {

            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)=img_in_.at<Vec3b>(i,j);
                }
            }
            w++;
            for(int i=h-2;i<=h+2;i++)
            {
                for(int j=w-2;j<=w+2;j++)
                {
                    img_with_map_.at<Vec3b>(i,j)[0]=0;
                    img_with_map_.at<Vec3b>(i,j)[1]=255;
                    img_with_map_.at<Vec3b>(i,j)[2]=0;
                }
            }
        }
        if(c=='b' )
        {
            cout<<x<<" "<<y<<" "<<w<<" "<<h<<endl;
        }
        c=waitKey();
    }
}


Vector3i map_drawer::color(float v)
{
    Vector3i vec;
    vec<<
          0,0,0;
    /*if(v>0.5)
    {
        float d=v-0.5;
        int r=(d/0.5)*255;
        if(r>255)
        {
            r=255;
        }
        if(r<0)
        {
            r=0;
        }
        vec(2)=r;
    }
    if(v>0.3 && v<0.6)
    {
        float d=v-0.3;
        int g=(d/0.5)*255;
        if(g>255)
        {
            g=255;
        }
        if(g<0)
        {
            g=0;
        }
        vec(1)=g;
    }
    if( v<0.4)
    {
        float d=v;
        int b=(d/0.4)*255;
        if(b>255)
        {
            b=255;
        }
        if(b<0)
        {
            b=0;
        }
        vec(0)=b;
    }*/
    vec(0)=(1-v)*255;
    vec(1)=(0.3-fabs(v-0.5))*255;
    vec(2)=v*255;
    for(int i=0;i<3;i++)
    {
        if(vec(i)<0)
        {
            vec(i)=0;
        }
        if(vec(i)>255)
        {
            vec(i)=255;
        }
    }

    //cout<<v<<endl;
    return vec;
}
