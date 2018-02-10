package com.company;

/**
 * Created by DELL on 2017/11/22.
 */
public class Point {
    int x;
    int y;
    int z;
    int gCost;
    int hEstimate;
    int fTotal;
    //double penty;
    Point prev;

    public String getKey(){
        return x+","+y+","+z;
    }
    public Point(int x, int y) {
        super();
        this.x = x;
        this.y = y;
    }


    public Point(int x, int y, int z) {
        super();
        this.x = x;
        this.y = y;
        this.z = z;
    }


    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + x;
        result = prime * result + y;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Point other = (Point) obj;
        if (x != other.x)
            return false;
        if (y != other.y)
            return false;
        return true;
    }
}
