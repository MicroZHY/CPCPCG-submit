#include <fstream>
#include <vector>
#include <bits/stdc++.h>
// COO格式的矩阵元素结构体

#define eps 1e-8
struct COOElement
{
    int row;    // 行索引
    int col;    // 列索引
    double val; // 值
};

using namespace std;

// 读取文件并将矩阵保存为COO格式
void readMatrixFile(const string &filename, vector<int> &rowval, vector<int> &colval,
                    vector<double> &valval)
{

    ifstream fin(filename);
    if (!fin)
    {
        cerr << "Error: cannot open file " << filename << endl;
        return;
    }

    int nRows, nCols, nNonZeros;
    while (fin >> nRows >> nNonZeros)
    {
        vector<int> cols(nNonZeros);
        vector<double> values(nNonZeros);
        for (int i = 0; i < nNonZeros; i++)
        {
            fin >> cols[i];
        }
        for (int i = 0; i < nNonZeros; i++)
        {
            fin >> values[i];
        }

        for (int i = 0; i < nNonZeros; i++)
        {
            if (fabs(values[i]) > eps)
            {
                rowval.push_back(nRows);
                colval.push_back(cols[i]);
                valval.push_back(values[i]);
            }
        }
    }

    fin.close();
}

// 将COO格式的矩阵保存为MTX文件
void save_coo_as_mtx(const std::vector<COOElement> &coo, int nrows, int ncols, const std::string &filename)
{
    // 打开输出文件流，将数据保存到文件中
    std::ofstream outfile(filename, std::ios::binary);

    // 写入MTX文件头
    outfile << "%%MatrixMarket matrix coordinate real general\n";
    outfile << nrows << ' ' << ncols << ' ' << coo.size() << endl;

    // 写入COO格式的矩阵元素
    for (const auto &elem : coo)
    {
        outfile << elem.row + 1  << ' ' << elem.col + 1  << ' ' << elem.val << endl;
    }

    // 关闭输出文件流
    outfile.close();
}

int main(int argc, char **argv)
{
    // 示例COO格式的矩阵
    std::vector<COOElement> coo;

    string filename1 = argv[1];
    string filename2 = argv[2];
    string filename3 = argv[3];


    std::ifstream infile1(filename1);
    std::ifstream infile2(filename2);

    std::vector<double> vec1, vec2;
    double temp;
    int faces=0;
    while (infile1 >> temp)
    {   
        faces++;
        vec1.push_back(temp);
    }

    cout<<"faces = "<< faces  <<endl;

    while (infile2 >> temp)
    {
        vec2.push_back(temp);
    }

    int cells =  *std::max_element(vec1.begin(), vec1.end())+1;

    // readMatrixFile(filename, rowval, colval, valval);
    // auto max_it = std::max_element(rowval.begin(), rowval.end());
    // int n = *max_it + 1 - 1105920;
    // int nnz = rowval.size();

    for (int i = 0; i < faces; i++)
    {
        COOElement elem;
        elem.row = vec1[i];
        elem.col = vec2[i];
        elem.val = 1;
        coo.push_back(elem);
    }


     for (int i = 0; i < faces; i++)
    {
        COOElement elem;
        elem.row = vec2[i];
        elem.col = vec1[i];
        elem.val = 1;
        coo.push_back(elem);
    }

      for (int i = 0; i < cells; i++)
    {
        COOElement elem;
        elem.row = i;
        elem.col = i;
        elem.val = 1;
        coo.push_back(elem);
    }

    std::string outname = filename3 + ".mtx";

    // 将COO格式的矩阵保存为MTX文件
    save_coo_as_mtx(coo, cells, cells, outname);

    return 0;
}

// #include <iostream>
// #include <fstream>
// #include <vector>

// int main()
// {
//     std::ifstream infile1("vector1.txt");
//     std::ifstream infile2("vector2.txt");

//     std::vector<double> vec1, vec2;
//     double temp;

//     while (infile1 >> temp)
//     {
//         vec1.push_back(temp);
//     }

//     while (infile2 >> temp)
//     {
//         vec2.push_back(temp);
//     }

//     std::cout << "Vector 1: ";
//     for (auto i : vec1)
//     {
//         std::cout << i << " ";
//     }
//     std::cout << std::endl;

//     std::cout << "Vector 2: ";
//     for (auto i : vec2)
//     {
//         std::cout << i << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }
