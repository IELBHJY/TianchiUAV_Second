package com.company;

import com.csvreader.CsvReader;

import java.nio.charset.Charset;
import java.util.*;

/**
 * Created by apple on 2018/2/6.
 */
public class GA {
    public class GAPoint{
        int number;
        int flyTime;
        public GAPoint(int number1,int flyTime){
            this.number=number1;
            this.flyTime=flyTime;
        }
    }
    private final int city;
    private final int popsize;
    public int[] xCity=new int[10];
    public int[] yCity=new int[10];
    public int[] endCitys=new int[3463];
    public int[] flyTimes=new int[3463];
    public double[] risks=new double[3463];
    private AStar aStar;
    public Set<Point> barrier = new HashSet<Point>();
    public double bestscore=0;
    public GAPoint[] bestPoints;
    public GA(int city,int popsize,int day){
        this.city=city;
        this.popsize=popsize;
        bestPoints=new GAPoint[city];
        //aStar=new AStar(day);
    }

    public List<GAPoint[]> childPoints=new ArrayList<>();
    public List<GAPoint[]> parentPoints=new ArrayList<>();

    /**
     * 设定边界限制
     */
    public void readBarrier(){
        for(int i=0;i<548;i++){
            barrier.add(new Point(i,-1));
        }
        for(int i=0;i<421;i++)
        {
            barrier.add(new Point(-1,i));
        }
        for(int i=0;i<548;i++){
            barrier.add(new Point(i,421));
        }
        for(int i=0;i<421;i++){
            barrier.add(new Point(548,i));
        }
        barrier.add(new Point(-1,-1));
        barrier.add(new Point(-1,421));
        barrier.add(new Point(548,421));
        barrier.add(new Point(548,-1));
    }

    /**
     * 读取城市坐标
     */
    public void readCity(){
        try {
            int m = 0;
            while (m < 12){
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Desktop/CityData.csv";
                CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("UTF-8"));
                while (reader.readRecord()){
                    csvFileList.add(reader.getValues());
                }
                reader.close();
                for (int i = 0; i<10; i++) {
                    String[] strData = csvFileList.get(i+2);
                    xCity[i]=Integer.parseInt(strData[1]);
                    yCity[i]=Integer.parseInt(strData[2]);
                }
                m++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void readdate(int day){
        try {
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Desktop/routes1_multi/"+String.valueOf(day)+"_risk.csv";
                CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("UTF-8"));
                while (reader.readRecord()){
                    csvFileList.add(reader.getValues());
                }
                reader.close();
                for (int i = 0; i<3463; i++) {
                    String[] strData = csvFileList.get(i+1);
                    endCitys[i]=Integer.parseInt(strData[0]);
                    flyTimes[i]=Integer.parseInt(strData[1]);
                    risks[i]=Double.parseDouble(strData[2]);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * 初始化解，解是0-9的序列，表示起飞顺序，此算法还未加起飞时间的搜索
     * @param n
     */
    public  void inital(int n){
        for(int i=0;i<n;i++){
            GAPoint[] temp=new GAPoint[city];
            List<Integer> list=new ArrayList<>();
            List<Integer> flylist=new ArrayList<>();
            int count=0;
            while(count<city) {
                int number=(int) (Math.random() * 10);
                if (!list.contains(number)) {
                    list.add(number);
                    while(true) {
                        int fly = (int) (Math.random() * 476);
                        if(!flylist.isEmpty()){
                            int isRight=-1;
                            for(Integer f:flylist){
                                if(Math.abs(fly-f)<5){
                                    isRight=1;
                                    break;
                                }
                            }
                            if(isRight==-1){
                                flylist.add(fly);
                                GAPoint p = new GAPoint(number, fly);
                                temp[count] = p;
                                break;
                            }
                        }
                        else{
                            flylist.add(fly);
                            GAPoint p = new GAPoint(number, fly);
                            temp[count] = p;
                            break;
                        }
                    }
                    count++;
                }
            }
            //System.out.println(i);
            parentPoints.add(temp);
        }
    }

    /**
     * 调用ASter 算法求解 每种起飞顺序下的飞行时间总和，若没有求出路径，当成1440。
     * @param list  传入父代
     * @return
     */
    public double[] calFlyTime(List<GAPoint[]> list){
        double[] ans=new double[popsize];
        long start = System.currentTimeMillis();
        for(int i=0;i<popsize;i++){
            double sum=0;
            for(int j=0;j<city;j++){
                Point endPoint;
                List<Point> set =new ArrayList<>();
                aStar.move(new Point(141,327,list.get(i)[j].flyTime), xCity[list.get(i)[j].number]-1, yCity[list.get(i)[j].number]-1,barrier);
                endPoint = aStar.getEndPoint();
                set = get(endPoint, set);
                //System.out.println("飞行时间："+(set.size()-1)*2);
                if(set.size()==1){
                    sum+=1440;
                }
                else{
                    sum+=(set.size()-1)*2;
                }
            }
            System.out.println("个体"+String.valueOf(i)+" "+"飞行时间总和是："+sum);
            ans[i]=sum;
        }
        long end = System.currentTimeMillis();
        System.out.println("求解用时"+(end - start)/1000+"秒");
        return ans;
    }

    public double[] calFlyTime1(List<GAPoint[]> list){
        double[] ans=new double[popsize];
        //double[] ans1=new double[popsize];
        //long start = System.currentTimeMillis();
        for(int i=0;i<popsize;i++){
            //double sum=0;
            int count=0;
            double min=Double.MAX_VALUE;
            for(int j=0;j<city;j++){
                for(int z=0;z<endCitys.length;z++){
                    if(endCitys[z]==list.get(i)[j].number+1&&flyTimes[z]==list.get(i)[j].flyTime){
                        //sum+=risks[z];
                        count++;
                        if(risks[z]<min){
                            min=risks[z];
                        }
                        break;
                    }
                }
            }
            if(count!=10){
                ans[i]=0;
            }
            else{
                ans[i]=min;
            }
            //System.out.println("count:"+count);
            //System.out.println("个体"+String.valueOf(i)+" "+"飞行时间总和是："+sum);
            //ans[i]=min;
            //ans[i]=sum;
        }
        //long end = System.currentTimeMillis();
        //System.out.println("求解用时"+(end - start)/1000+"秒");
        return ans;
    }
    /**
     * 求解路径长度
     * @param p
     * @param set
     * @return
     */
    public List<Point> get(Point p, List<Point> set) {
        if (p != null) {
            set.add(p);
        }
        Point pp = p.prev;
        if (pp != null) {
            get(pp, set);
        } else {
            return set;
        }
        return set;
    }

    /**
     * 运用精英保留策略和轮盘赌策略，确定哪些父代个体参与下一代进化，将标号存储在 返回数组中，顺序是：先精英，再交叉个体，再变异个体
     * @param scores
     * @param rate1
     * @param rate2
     * @param rate3
     * @return
     */
    public int[] select(double[] scores,double rate1,double rate2,double rate3){
        int[] ans=new int[popsize];
        List<Double> temp=new ArrayList<>();
        int count=0;
        int label=0;
        while(count<popsize*rate3){
            double d=Double.MIN_VALUE;
            for(int i=0;i<popsize;i++){
                if(!temp.contains(scores[i])&&scores[i]>d){
                    d=scores[i];
                    label=i;
                }
            }
            temp.add(d);
            ans[count++]=label;
        }
        //保存最好的个体
        double max =Collections.max(temp);
        //System.out.println(scores[number]);
        if(max>bestscore) {
            bestscore=max;
            int number=temp.indexOf(max);
            GAPoint[] best = parentPoints.get(ans[number]);
            for (int i = 0; i < best.length; i++) {
                bestPoints[i] = new GAPoint(best[i].number,best[i].flyTime);
            }
        }
        System.out.println(bestscore);
        double sum=0;
        for(int i=0;i<popsize;i++){
            sum+=scores[i];
        }
        int count1=0;
        while(count1<popsize*rate1+popsize*rate2){
            double rand=(Math.random() * sum);
            double sum1=0;
            for(int i=0;i<popsize;i++){
                sum1+=scores[i];
                if(sum1>=rand){
                    ans[count++]=i;
                    break;
                }
            }
            count1++;
        }
        return ans;
    }

    /**
     *  精英保留
     * @param p
     */
    public void inheritant(int p){
        GAPoint[] temp=parentPoints.get(p);
        GAPoint[] temp1=new GAPoint[temp.length];
        for(int i=0;i<temp.length;i++){
            temp1[i]=new GAPoint(temp[i].number,temp[i].flyTime);
        }
        childPoints.add(temp1);
    }

    /**
     * 交叉
     * @param p1
     * @param p2
     */
    public void crossover(int p1,int p2){
        GAPoint[] temp=parentPoints.get(p1);
        GAPoint[] temp1=new GAPoint[temp.length];
        for(int i=0;i<temp.length;i++){
            temp1[i]=new GAPoint(temp[i].number,temp[i].flyTime);
        }
        GAPoint[] temp3=parentPoints.get(p2);
        GAPoint[] temp2=new GAPoint[temp3.length];
        for(int i=0;i<temp3.length;i++){
            temp2[i]=new GAPoint(temp3[i].number,temp3[i].flyTime);
        }
        //选择随机点交叉
        int position=(int)(Math.random()*10);
        List<Integer> listP1=new ArrayList<>();
        List<Integer> listP2=new ArrayList<>();
        for(int i=position;i<temp1.length;i++){
            listP1.add(temp1[i].number);
            listP2.add(temp2[i].number);
        }
        List<GAPoint> list1=new ArrayList<>();
        List<GAPoint> list2=new ArrayList<>();
        for(int i=0;i<temp2.length;i++){
            if(listP1.contains(temp2[i].number)){
                list1.add(temp2[i]);
            }
        }
        for(int i=0;i<temp1.length;i++){
            if(listP2.contains(temp1[i].number)){
                list2.add(temp1[i]);
            }
        }
        GAPoint[] newtemp1=new GAPoint[temp1.length];
        for(int i=0;i<city;i++){
            newtemp1[i]=new GAPoint(temp1[i].number,temp1[i].flyTime);
        }
        int label=0;
        for(int i=position;i<temp1.length;i++){
            GAPoint top=list1.get(label++);
            newtemp1[i].number=top.number;
        }
        GAPoint[] newtemp2=new GAPoint[temp2.length];
        for(int i=0;i<city;i++){
            newtemp2[i]=new GAPoint(temp2[i].number,temp2[i].flyTime);
        }
        label=0;
        for(int i=position;i<temp2.length;i++){
            GAPoint top=list2.get(label++);
            newtemp2[i].number=top.number;
        }
        childPoints.add(newtemp1);
        childPoints.add(newtemp2);
    }

    /**
     * 变异
     * @param p
     */
    public void mutation(int p){
        GAPoint[] temp=parentPoints.get(p);
        GAPoint[] temp1=new GAPoint[temp.length];
        for(int i=0;i<temp.length;i++){
            temp1[i]=new GAPoint(temp[i].number,temp[i].flyTime);
        }
        //随机选择变异点
        int position1=(int)(Math.random()*10);
        int position2=(int) (Math.random()*10);
        while(position2==position1){
            position2=(int) (Math.random()*10);
        }
        int number=temp1[position1].number;
        temp1[position1].number=temp1[position2].number;
        temp1[position2].number=number;
        childPoints.add(temp1);
    }

    /**
     * 将下一代个体付给父代，继续下一次进化
     */
    public void childToParent(){
        parentPoints=new ArrayList<>();
        for(int i=0;i<popsize;i++){
            GAPoint[] temp=childPoints.get(i);
            GAPoint[] temp1=new GAPoint[temp.length];
            for(int j=0;j<temp.length;j++){
                temp1[j]=temp[j];
            }
            parentPoints.add(temp1);
        }
    }
}
