package com.company;

import com.csvreader.CsvReader;

import java.nio.charset.Charset;
import java.util.*;

/**
 * Created by DELL on 2017/11/22.
 */
public class AStar {
    public static final int MOVE_TETN = 2;
    public static final int LENGHT = 2;
    private static final int totalRow = 548;
    private static final int totalCom = 421;
    int label=0;
    public static Double[][][] weather = new Double[18][totalRow][totalCom];
    public static Double[][][] rainfall=new Double[18][totalRow][totalCom];
    /* 打开的列表 */
    Map<String, Point> openMap = new HashMap<String, Point>();
    /* 关闭的列表 */
    Map<String, Point> closeMap = new HashMap<String, Point>();
    /* 障碍物 */
    Set<Point> barrier;
    /* 起点 */
    public Point startPoint;
    /* 终点 */
    public Point endPoint;
    /* 当前使用节点 */
    Point currentPoint;
    /* 循环次数，为了防止目标不可到达 */
    int num = 0;
    public AStar(int day){
        readData(day);
        readRainFall(day);
        System.out.println("风速值和降雨量读取完成！");
    }
    public void move(Point s, int x2, int y2,Set<Point> barrier1) {
        num = 0;
        startPoint = new Point(s.x,s.y,s.z);
        while(weather[startPoint.z/30][startPoint.x][startPoint.y]>=0.27||rainfall[startPoint.z/30][startPoint.x][startPoint.y]>=4){
            startPoint.z++;
        }
        endPoint=new Point(x2,y2);
        closeMap.put(startPoint.getKey(), startPoint);
        currentPoint = startPoint;
        barrier=barrier1;
        Point p=currentPoint;
        toOpen(p.x,p.y);
        while(num<50000) {
            p = toClose();
            if (p.x!=endPoint.x||p.y!=endPoint.y) {
                toOpen(p.x, p.y);
            }
            else{
                endPoint=p;
                break;
            }
            num++;
        }
        if(p.equals(endPoint)){
            System.out.println("找到路径！");
        }
        else{
            System.out.println("未找到路径，找距离终点理论最近点");
        }
    }
    public static void readData(int day)
    {
        try {
            int m = 0;
            while (m < 18){
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Desktop/lgb_pre20180208/Testday"+String.valueOf(day)+"_hour"+ String.valueOf(m + 3) + ".csv";
                CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("UTF-8"));
                while (reader.readRecord()){
                    csvFileList.add(reader.getValues());
                }
                reader.close();
                for (int i = 0; i<totalRow; i++) {
                    String[] strData = csvFileList.get(i);
                    for (int j = 0; j< totalCom; j++){
                        double tmp =Double.valueOf(strData[j]);
                        weather[m][i][j] = tmp;
                    }
                }
                m++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public static void readRainFall(int day)
    {
        try {
            int m = 0;
            while (m < 18){
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Desktop/20180205rainfall/Testday"+String.valueOf(day)+"_hour"+ String.valueOf(m + 3) + ".csv";
                CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("UTF-8"));
                while (reader.readRecord()){
                    csvFileList.add(reader.getValues());
                }
                reader.close();
                for (int i = 0; i<totalRow; i++) {
                    String[] strData = csvFileList.get(i);
                    for (int j = 0; j< totalCom; j++){
                        double tmp =Double.valueOf(strData[j]);
                        rainfall[m][i][j] = tmp;
                    }
                }
                m++;
            }
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
    private int getGuessLength(int x1, int y1, int x2, int y2) {
        //return ((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 -y1))* AStar.LENGHT;
        return (Math.abs(x1 - x2) + Math.abs(y1 - y2)) * AStar.LENGHT;
        //return Math.max(Math.abs(x1 - x2), Math.abs(y1 - y2)) * AStar.LENGHT;
        //return 0;
    }


    private void toOpen(int x, int y) {
        addOpenPoint(new Point(x - 1, y,currentPoint.z+1), AStar.MOVE_TETN);
        addOpenPoint(new Point(x + 1, y,currentPoint.z+1), AStar.MOVE_TETN);
        addOpenPoint(new Point(x, y - 1,currentPoint.z+1), AStar.MOVE_TETN);
        addOpenPoint(new Point(x, y + 1,currentPoint.z+1), AStar.MOVE_TETN);
        addOpenPoint(new Point(x,y,currentPoint.z+1),AStar.MOVE_TETN);//停留点
    }
    private void addOpenPoint(Point point, int gCost) {
        if (point.x < 0 || point.y < 0) {
            return;
        }
        String key = point.getKey();
        //如果待搜索的点不包含在barrier内，并且该点距离起始点飞行时间小于18小时,weather中存放位置的天气概率，即风速值越小，概率越小。
        //将周围的点均加入openlist，然后概率值大那么惩罚高，这样路径就会优先选择风速值超级低的路径，一定程度上限制了踩雷的概率。
        Point temp=new Point(point.x,point.y);
        if(!barrier.contains(temp)) {
            int moveLength=point.z;//根据z坐标计算飞行时间
            int hour=moveLength/30;
            if(hour<=17){
                if(weather[hour][point.x][point.y]>=0.27||rainfall[hour][point.x][point.y]>=4) {
                    return;
                }
            }else{
                return;
            }
            /*double wind=weather[hour][point.x][point.y];
            Double p=20*wind;
            int penty=p.intValue();
            double rain=rainfall[hour][point.x][point.y];
            Double p1=20*rain;
            int penty1=p1.intValue();*/
            int hEstimate = getGuessLength(point.x, point.y, endPoint.x, endPoint.y);
            int totalGCost = currentPoint.gCost + gCost;
            int fTotal = totalGCost + hEstimate;
            if (!closeMap.containsKey(key)) {
                point.hEstimate = hEstimate;
                point.gCost = totalGCost;
                point.fTotal = fTotal;
                Point oldPoint = openMap.get(key);
                if (oldPoint != null) {
                    if (oldPoint.gCost > totalGCost) {
                        oldPoint.fTotal = fTotal;
                        oldPoint.prev = currentPoint;
                        openMap.put(key, oldPoint);
                    }
                } else {
                    point.prev = currentPoint;
                    openMap.put(key, point);
                }
            }
        }
    }

    private Point toClose() {
        List<Point> list = new ArrayList<>(openMap.values());
        Collections.sort(list, new Comparator<Point>() {
            @Override
            public int compare(Point o1, Point o2) {
                if (o1.fTotal > o2.fTotal) {
                    return 1;
                } else if (o1.fTotal < o2.fTotal) {
                    return -1;
                } else {
                    return 0;
                }
            }
        });
        if (list.size() > 0) {
            currentPoint = list.get(0);
            closeMap.put(currentPoint.getKey(), currentPoint);
            openMap.remove(currentPoint.getKey());
        }
        return currentPoint;
    }

    public Point getEndPoint() {
        return endPoint;
    }

    public Point getStartPoint()
    {
        return startPoint;
    }

    public double moveAway(Point p,int hour){
        double def=Double.MAX_VALUE;
        for(int i=0;i<548;i++){
            for(int j=0;j<421;j++){
                if(weather[hour][i][j]>=1){
                    double distance=getGuessLength(p.x,p.y,i,j);
                    if(def>distance){
                        def=distance;
                    }
                }
            }
        }
        return def;
    }

    public double getEstimate(Point point){
       double result=(Math.abs(point.x-endPoint.x) + Math.abs(point.y - endPoint.y)) * AStar.LENGHT;
       int count=0;
       for(int i=0;i<548;i++){
           for(int j=0;j<421;j++){
               if(i<Math.min(point.x,endPoint.x)||i>Math.max(point.x,endPoint.x)||j<Math.min(point.y,endPoint.y)||j>Math.max(point.y,endPoint.y)) {
                   continue;
               }
           }
       }
       return 0.0;
    }
}
