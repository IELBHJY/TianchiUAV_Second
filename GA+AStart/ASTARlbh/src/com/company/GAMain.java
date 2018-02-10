package com.company;

import com.csvreader.CsvWriter;

import java.io.IOException;
import java.nio.charset.Charset;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Created by apple on 2018/2/6.
 */
public class GAMain {
    public static void main(String[] args) {
        int popsize=10000;
        int city=10;
        int day=10;
        double rate1=0.4;
        double rate2=0.4;
        double rate3=0.2;
        int MAX_iteration=40;
        GA ga=new GA(city,popsize,day);
        ga.readdate(day);
        ga.inital(popsize);
        int iteration=0;
        while(iteration<MAX_iteration){
            System.out.println(iteration);
            double[] scores=ga.calFlyTime1(ga.parentPoints);
            int[] labels=ga.select(scores,rate1,rate2,rate3);
            ga.childPoints=new ArrayList<>();
            for(int i=0;i<(int)(popsize*rate3);i++){
                ga.inheritant(labels[i]);
            }
            for(int i=(int)(popsize*rate3);i<(int)(popsize*rate3+popsize*rate1);i=i+2){
                ga.crossover(labels[i],labels[i+1]);
            }
            for(int i=(int)(popsize*rate3+popsize*rate1);i<popsize;i++){
                ga.mutation(labels[i]);
            }
            ga.childToParent();
            iteration++;
        }
        double sum=0;
        double[] cityScores=new double[city];
        for(int j=0;j<city;j++){
            for(int z=0;z<ga.endCitys.length;z++){
                if(ga.endCitys[z]==ga.bestPoints[j].number+1&&ga.bestPoints[j].flyTime==ga.flyTimes[z]){
                    sum+=ga.risks[z];
                    cityScores[j]=ga.risks[z];
                    System.out.println(ga.bestPoints[j].number+"  "+ga.bestPoints[j].flyTime+" "+ga.risks[z]);
                }
            }
        }
        System.out.println("总分数是："+sum);
        write(ga.bestPoints,cityScores,day);
    }
    private static void writeData(List<Point> set,int cityNumber,int Data){
        try {
            String csvFilePath = "/Users/apple/Desktop/result0205/"+String.valueOf(Data)+"/"+ String.valueOf(cityNumber)+ ".csv" ;
            // 创建CSV写对象 例如:CsvWriter(文件路径，分隔符，编码格式);
            CsvWriter csvWriter = new CsvWriter(csvFilePath, ',', Charset.forName("UTF-8"));
            SimpleDateFormat df=new SimpleDateFormat("HH:mm");
            Calendar calendar=new GregorianCalendar(2018,1,19,2,58,0);
            calendar.add(Calendar.MINUTE,(cityNumber-1)*10);
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
                String[] ordination=set.get(set.size()-1-i).getKey().split(",");
                csvContent[3]=String.valueOf(Integer.parseInt(ordination[0])+1);
                csvContent[4]=String.valueOf(Integer.parseInt(ordination[1])+1);
                csvWriter.writeRecord(csvContent);
            }
            csvWriter.close();
            System.out.println("------csv文件已经写入---------");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void write(GA.GAPoint[] bestPoints,double[] bestscore,int day){
        try {
            String csvFilePath = "/Users/apple/Desktop/result0208/"+ String.valueOf(day)+ ".csv" ;
            // 创建CSV写对象 例如:CsvWriter(文件路径，分隔符，编码格式);
            CsvWriter csvWriter = new CsvWriter(csvFilePath, ',', Charset.forName("UTF-8"));
            String[] table={"city","flyTime","scores"};
            csvWriter.writeRecord(table);
            for(int i=0;i<bestPoints.length;i++){
                String[] csvContent = new String[3];
                csvContent[0]=String.valueOf(bestPoints[i].number);
                csvContent[1]=String.valueOf(bestPoints[i].flyTime);
                csvContent[2]=String.valueOf(bestscore[i]);
                csvWriter.writeRecord(csvContent);
            }
            csvWriter.close();
            System.out.println("------csv文件已经写入---------");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
