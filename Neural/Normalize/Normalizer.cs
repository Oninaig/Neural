using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Markup;

namespace Normalize
{
    public static class Normalizer
    {
        public static void GuassNormal(double[][] data, int column)
        {
            int j = column;
            double sum = 0.0;
            for (int i = 0; i < data.Length; ++i)
                sum += data[i][j];
            double mean = sum / data.Length;

            double sumSquares = 0.0;
            for (int i = 0; i < data.Length; ++i)
                sumSquares += (data[i][j] - mean) * (data[i][j] - mean);
            double stdDev = Math.Sqrt(sumSquares / data.Length);

            for (int i = 0; i < data.Length; ++i)
                data[i][j] = (data[i][j] - mean) / stdDev;
        }

        public static void MinMaxNormal(double[][] data, int column)
        {
            int j = column;
            double min = data[0][j];
            double max = data[0][j];
            for (int i = 0; i < data.Length; ++i)
            {
                if (data[i][j] < min)
                    min = data[i][j];
                if (data[i][j] > max)
                    max = data[i][j];
            }

            double range = max - min;
            if (range == 0.0) // ugly. All values are the same
            {
                for (int i = 0; i < data.Length; ++i)
                {
                    data[i][j] = 0.5;
                }
                return;
            }

            for (int i = 0; i < data.Length; ++i)
            {
                data[i][j] = (data[i][j] - min) / range;
            }
        }

        public static void ShowMatrix(double[][] matrix, int decimals)
        {
            for (int i = 0; i < matrix.Length; ++i)
            {
                for (int j = 0; j < matrix[i].Length;++j)
                {
                    double v = Math.Abs(matrix[i][j]);
                    if (matrix[i][j] >= 0.0)
                        Console.Write(" ");
                    else
                        Console.Write("-");
                    Console.Write(v.ToString("F" + decimals).PadRight(5) + " ");
                }
                Console.WriteLine("");
            }
        }

        public static void ShowData(string[] rawData)
        {
            for(int i =0; i < rawData.Length; i++)
            {
                Console.WriteLine(rawData[i]);
            }
            Console.WriteLine("");
        }

        public static void EncodeFile(string originalFile, string encodedFile, int column, string encodingType)
        {
            //encodingType: "effects" or "dummy"
            FileStream ifs = new FileStream(originalFile, FileMode.Open);
            StreamReader sr = new StreamReader(ifs);
            string line = "";
            string[] tokens = null;

            //Create dictionary of distinct items in the column
            Dictionary<string, int> d = new Dictionary<string, int>();
            int itemNum = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(','); //Assumes items are comma-delimited
                if (d.ContainsKey(tokens[column]) == false)
                    d.Add(tokens[column], itemNum++);
            }
            sr.Close();
            ifs.Close();


            //Write the result fle
            int N = d.Count;
            ifs = new FileStream(originalFile, FileMode.Open);
            sr = new StreamReader(ifs);
            FileStream ofs = new FileStream(encodedFile, FileMode.Create);
            StreamWriter sw = new StreamWriter(ofs);
            string s = null; // result line

            while ((line = sr.ReadLine()) != null)
            {
                s = "";
                tokens = line.Split(',');

                //if the current token is not in the target column, it is added as-is.
                //if the current token is in the target column, it is replaced by the appropriate encoding.
                for (int i = 0; i < tokens.Length; i++)
                {
                    if (i == column)
                    {
                        //Encode this string
                        int index = d[tokens[i]]; // 0, 1, 2, or ...
                        if (encodingType == "effects")
                            s += EffectsEncoding(index, N) + ",";
                        else if (encodingType == "dummy")
                            s += DummyEncoding(index, N) + ",";

                    }
                    else
                    {
                        s += tokens[i] + ",";
                    }
                }
                s.Remove(s.Length - 1); //remove trailing ','.
                sw.WriteLine(s); // write the string to the file.
            }

            sw.Close(); ofs.Close();
            sr.Close();
            ifs.Close();

        }

        public static string EffectsEncoding(int index, int N)
        {
            //Binary value
            if (N == 2)
            {
                if (index == 0) return "-1";
                else if (index == 1) return "1";
            }

            int[] values = new int[N - 1];
            if (index == N - 1)
            {
                //Last item is all -1s
                for (int i = 0; i < values.Length; i++)
                    values[i] = -1;
            }
            else
            {
                values[index] = 1; // 0 values are already there on array init
            }

            string s = values[0].ToString();
            for (int i = 1; i < values.Length; i++)
            {
                s += "," + values[i];
            }
            return s;
        }

        public static string DummyEncoding(int index, int N)
        {
            int[] values = new int[N];
            values[index] = 1;

            string s = values[0].ToString();
            for (int i = 1; i < values.Length; i++)
                s += "," + values[i];
            return s;
        }

    }
}
