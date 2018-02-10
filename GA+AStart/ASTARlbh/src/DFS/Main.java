package DFS;

import com.csvreader.CsvReader;
import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by apple on 2018/1/25.
 */
public class Main {

    private static final int m = 548;
    private static final int n = 421;
    private static int xCity[]=new int[10];
    private static int yCity[]=new int[10];
    public static double weather[][][]=new double[18][m][n];
    public static void main(String args[]){
        readCity();
        for(int i=6;i<11;i++){
            readData(i);
            for(int j=3;j<4;j++){
                DFS(i,j);
            }
        }
        /*Stack<Node> stkRev = new Stack<>();//将路径反过来，因为栈中输出的路径是反的
        List<Node> sets=new ArrayList<>();
        while (!stack.isEmpty()) {
            sets.add(stack.pop());
        }*/
        /*while (!stkRev.isEmpty()) {
            System.out.println("(" + stkRev.peek().x + "," + stkRev.peek().y +","+ stkRev.peek().z+ ")");
            stkRev.pop();
        }*/
        /*for(Node node:sets){
            System.out.println("(" + node.x + "," + node.y +","+ node.z+ ")");
        }*/
    }

    private static void DFS(int day,int city){
        int[][] dir = {{1, 0}, {-1, 0},{0,1},{0,-1},{0,0}};//定义两个方向横着走或者竖着走（题目中说只走这两个方向，当前也可以定义多个方向）
        Stack<Node> stack = new Stack<>();//定义一个栈，保存路径
        List<Node> sets=new ArrayList<>();
        Node start = new Node(141, 327,0);//定义起始位置
        Node end = new Node(xCity[city-1]-1,yCity[city-1]-1 ,540);//定义目的位置
        stack.push(start);//将起始点加入队列
        while (!stack.isEmpty()) {//如果stack为空了还没有找到解，说明就没有通路，当然本题不存在无解，题目上说了一定存在一个通路。
            boolean flag = false;//标记是否找了一个方向
            Node pek = stack.peek();//获取栈顶元素，注意不需要出栈
            if (pek.x == end.x && pek.y == end.y) {//如果到达目的地则跳出循环
                while (!stack.isEmpty()) {
                    sets.add(stack.pop());
                }
                //writeData(sets,city,day);
                System.out.println("第"+day+"天，飞往城市"+city+"的路径找到。飞行时间："+(sets.size()-1)*2+"分钟。");
                break;
            } else {
                double nodesValue[]={-1,-1,-1,-1,-1};
                for (int i = 0; i < 5; i++) {
                    if((pek.z+1)/30>17){continue;}
                    if(weather[(pek.z+1)/30][pek.x+dir[i][0]][pek.y + dir[i][1]]==1) {
                        continue;
                    }
                    Node temp = new Node(pek.x + dir[i][0], pek.y + dir[i][1],pek.z+1);
                    nodesValue[i]=calNodeValue(temp)+calEndingValue(temp.x,temp.y,end.x,end.y);
                }
                double min=Double.MAX_VALUE;
                int label=-1;
                for(int j=0;j<5;j++){
                    if(nodesValue[j]>=0&&nodesValue[j]<min){
                        min=nodesValue[j];
                        label=j;
                    }
                }
                if(label!=-1){
                    stack.push(new Node(pek.x + dir[label][0], pek.y + dir[label][1], pek.z+1));
                    //System.out.println(pek.x + dir[label][0]+" "+pek.y + dir[label][1]);
                    flag=true;
                }
                if (flag) {//找到了方向，就不用执行下面的出栈，沿着这个方向一直搜下去
                    continue;
                }
                stack.pop();//如果两个方向都不能通过，则出栈。
            }
        }
        if(sets.isEmpty()) {
            System.out.println("第" + day + "天，飞往城市" + city + "的路径没有找到。");
        }
    }

    private static double calNodeValue(Node x){
        double result=Double.MAX_VALUE;
       for(int i=0;i<m;i++){
           for(int j=0;j<n;j++){
               if(weather[x.z/30][i][j]==1){
                   if(calEndingValue(i,j,x.x,x.y)<result){
                       result=calEndingValue(i,j,x.x,x.y);
                   }
               }
           }
       }
       return result;
    }

    private static int calEndingValue(int x1,int y1,int x2,int y2){
        return (Math.abs(x1 - x2) + Math.abs(y1 - y2));
    }

    private  static void readCity(){
        try {
            int m = 0;
            while (m < 12){
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Documents/data/CityData.csv";
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

    private static void writeData(List<Node> set, int cityNumber, int Data){
        try {
            String csvFilePath = "/Users/apple/Desktop/DFSresult/"+String.valueOf(Data)+"/"+ String.valueOf(cityNumber)+ ".csv" ;
            // 创建CSV写对象 例如:CsvWriter(文件路径，分隔符，编码格式);
            CsvWriter csvWriter = new CsvWriter(csvFilePath, ',', Charset.forName("UTF-8"));
            SimpleDateFormat df=new SimpleDateFormat("HH:mm");
            Calendar calendar=new GregorianCalendar(2018,1,19,2,58,0);
            for(int i=0;i<set.size();i++){
                String[] csvContent = new String[5];
                csvContent[0]=String.valueOf(cityNumber);
                csvContent[1]=String.valueOf(Data);
                calendar.add(Calendar.MINUTE,2);
                Date date = calendar.getTime();
                String str=df.format(date);
                String[] temp=str.split(":");
                if(temp[0].charAt(0)=='0'){
                    temp[0]=String.valueOf(temp[0].charAt(1));
                }
                csvContent[2]=temp[0]+":"+temp[1];
                csvContent[3]=String.valueOf(set.get(set.size()-1-i).x+1);
                csvContent[4]=String.valueOf(set.get(set.size()-1-i).y+1);
                csvWriter.writeRecord(csvContent);
            }
            csvWriter.close();
            System.out.println("------csv文件已经写入---------");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void readData(int day)
    {
        try {
            int m = 0;
            while (m < 18){
                ArrayList<String[]> csvFileList = new ArrayList<String[]>();
                String csvFilePath = "/Users/apple/Desktop/20180118/Testday"+day+"_hour"+ String.valueOf(m + 3) + ".csv";
                CsvReader reader = new CsvReader(csvFilePath, ',', Charset.forName("UTF-8"));
                while (reader.readRecord()){
                    csvFileList.add(reader.getValues());
                }
                reader.close();
                for (int i = 0; i<548; i++) {
                    String[] strData = csvFileList.get(i);
                    for (int j = 0; j<421; j++){
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
}
class Node{
    int x;
    int y;
    int z;
    Node(int x,int y,int z){
        this.x=x;
        this.y=y;
        this.z=z;
    }
}

