//
//  main.cpp
//
//  Created by liuzixing on 05/2/2018.
//  Copyright © 2018 liuzixing. All rights reserved.
//

#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <utility>
#include <fstream>
using namespace std;
const int max_x = 548;
const int max_y = 421;
const int limited_max_x = 473;
const int limited_min_x = 40;
const int limited_max_y = 421;
const int limited_min_y = 138;
const int limited_proba_threshold = -120;
const int max_h = 18;
const int max_date = 5;
const int max_flight = 10;
const int max_start_point = 108;
const int max_step = 540;
static double map_proba[max_date][max_x][max_y][max_h];
static double dp[max_x][max_y][max_step];
const double MINPROBA = -500000;
int dx[] = {-1,0,1,0,0};
int dy[] = {0,1,0,-1,0};
int city_location_x[] ={142,84,199,140,236,315,358,363,423,125,189};
int city_location_y[] ={328,203,371,234,241,281,207,237,266,375,274};
double possible_solution_proba[max_date][max_flight][max_start_point];
int possible_solution_step[max_date][max_flight][max_start_point];

void read_file(){
    FILE * fp;
    fp = fopen ("wind_rain_rf.txt", "r+");
    for (int i = 0;i < max_date;i++){
        cout << "in progress:" << i << endl;
        for (int j = 0; j < max_x;j++){
            for (int k = 0;k < max_y;k++)
                for (int l = 0;l < max_h;l++){
                    double tmp = 0;
                    fscanf(fp, "%lf", &tmp);
                    map_proba[i][j][k][l] = tmp;
                }
        }
    }
    cout << map_proba[0][100][101][5] << endl;
    cout << map_proba[3][200][201][10] << endl;
    cout << "read file done." << endl;
}

void dijkstra_fill(int date_id,int start_point=0,int end_step=max_step-1){
    int origin_x = city_location_x[0];
    int origin_y = city_location_y[0];
    
    for (int i = 0;i < max_x;i++)
        for (int j = 0;j < max_y;j++)
            for (int k = 0;k < max_step;k++)
                dp[i][j][k] = MINPROBA;
    
    int start_step = start_point * 5;
    dp[origin_x-1][origin_y-1][start_step] = map_proba[date_id][origin_x-1][origin_y-1][start_step/30];
    for (int step = start_step;step < end_step;step++){
        int next_h = (step + 1) / 30;
        for (int x = limited_min_x;x < limited_max_x;x++){
            for (int y = limited_min_y; y < limited_max_y;y++){
                if (dp[x][y][step] > MINPROBA){
                    //if (dp[x][y][step] < limited_proba_threshold) continue;
                    for (int i = 0;i < 5;i++){
                        int next_x = x + dx[i];
                        int next_y = y + dy[i];
                        if (next_x >= limited_min_x && next_x < limited_max_x && next_y >= limited_min_y && next_y < limited_max_y ) {
                            double next_safe_proba = dp[x][y][step] + map_proba[date_id][next_x][next_y][next_h];
                            if (next_safe_proba > dp[next_x][next_y][step+1]){
                                dp[next_x][next_y][step+1] = next_safe_proba;
                            }
                        }
                    }
                }
            }
        }
    }
    cout << "dijkstra fill done!" << start_point << "/" << 107 << endl;
}
vector<int> greedy_select_solution(int date_id){
    int res[max_flight];
    int order[max_flight] = {7,5,6,4,0,3,9,1,2,8};
    int visited[max_start_point];
    for (int i = 0;i < max_start_point;i++)
        visited[i] = 0;
    for (int i = 0;i < max_flight;i++){
        int select_start_point = -1;
        double select_start_point_proba = MINPROBA;
        for (int j = 0;j < max_start_point;j++){
            if (visited[j] == 0 && select_start_point_proba < possible_solution_proba[date_id][order[i]][j]){
                select_start_point_proba = possible_solution_proba[date_id][order[i]][j];
                select_start_point = j;
            }
        }
        cout << order[i] << " " << select_start_point_proba << " " << select_start_point << endl;
        res[order[i]] = select_start_point;
        if (select_start_point >= 0)
            visited[select_start_point] = 1;
    }
    //vector<int>(res)
    return vector<int>(res,res+max_flight);
}
void fill_possible_solution(int date_id,int start_point=0){
    for (int i = 1;i <= 10;i++) {
        double max_proba = MINPROBA;
        int target_x = city_location_x[i] - 1;
        int target_y = city_location_y[i] - 1;
        int best_step = 0;
        for (int step = start_point * 5;step <= max_step-1;step++){
            if (dp[target_x][target_y][step] > max_proba){
                best_step = step;
                max_proba = dp[target_x][target_y][step];
            }
        }
        possible_solution_proba[date_id][i-1][start_point] = max_proba;
        possible_solution_step[date_id][i-1][start_point] = best_step;
    }
}

pair<vector<string>,vector<double>> dijkstra(int date_id){
    vector<double> score_list = {};
    vector<string> solution_list = {};
    for (int start_point = 0;start_point < max_start_point;start_point++){
        dijkstra_fill(date_id,start_point,max_step-1);
        fill_possible_solution(date_id,start_point);
    }
    vector<int> start_points = greedy_select_solution(date_id);

    for (int i = 1;i <= max_flight;i++) {
        int selected_start_point = start_points[i-1];
        if (selected_start_point < 0) continue;
        int best_step = possible_solution_step[date_id][i-1][selected_start_point];
        if (best_step == 0) continue;
        dijkstra_fill(date_id,selected_start_point,max_step-1);
        int current_x = city_location_x[i] - 1;
        int current_y = city_location_y[i] - 1;
        score_list.push_back(possible_solution_proba[date_id][i-1][selected_start_point]);
        while (true){
            int t = 180 + best_step * 2;
            char time_string[5];
            sprintf(time_string, "%02d:%02d", t/60,t%60);
            char record[50];
            sprintf(record, "%d,%d,%s,%d,%d", i,date_id+6, time_string,current_x + 1, current_y+1);
            solution_list.push_back(record);
            int previous_x = current_x;
            int previous_y = current_y;
            int current_h = int(best_step / 30);
            if (best_step == selected_start_point * 5)
                break;
            for (int j = 0;j < 5;j++){
                int previous_x_tmp = current_x + dx[j];
                int previous_y_tmp = current_y + dy[j];
                if (previous_x_tmp < limited_max_x && previous_x_tmp >= limited_min_x && previous_y_tmp < limited_max_y && previous_y_tmp >= limited_min_y ){
                    if (abs(map_proba[date_id][current_x][current_y][current_h] + dp[previous_x_tmp][previous_y_tmp][best_step-1] - dp[current_x][current_y][best_step]) < 1e-9){
                        previous_x = previous_x_tmp;
                        previous_y = previous_y_tmp;
                        break;
                    }
                }
            }
            current_x = previous_x;
            current_y = previous_y;
            best_step -= 1;
        }
    }
    return make_pair(solution_list, score_list);
}

void init_parameter() {
    for (int i = 0;i < max_date;i++)
        for (int j = 0;j < max_flight;j++)
            for (int k = 0;k < max_start_point;k++){
                possible_solution_proba[i][j][k] = MINPROBA;
                possible_solution_step[i][j][k] = 0;
            }
}

int main(int argc, const char * argv[]) {
    init_parameter();
    read_file();
    vector<double> final_score_list = {};
    vector<string> final_solution_list = {};

    for (int date_id = 0;date_id < 5;date_id++){
        pair<vector<string>,vector<double>> tmp = dijkstra(date_id);
        final_score_list.insert(final_score_list.end(),tmp.second.begin(),tmp.second.end());
        final_solution_list.insert(final_solution_list.end(),tmp.first.begin(),tmp.first.end());
        
    }
    
    ofstream myfile;
    myfile.open ("rf_avg_wind_rain_greedy_opti.csv");
    for (int i = 0;i < final_solution_list.size();i++){
        myfile << final_solution_list[i] << endl;
    }
    myfile.close();
    double sum = 0;
    for (int i = 0; i < final_score_list.size();i++){
        sum += final_score_list[i];
    }
    cout << sum << endl;
    cout << final_solution_list.size() * 2 << endl;
    return 0;
}

