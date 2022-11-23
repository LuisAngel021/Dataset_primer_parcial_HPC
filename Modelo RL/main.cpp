/*
 * Fecha: 21/09/2022
 * Autor: Luis Ángel Rodriguez
 * Materia: HPC
 * Tópico: Implementación de la Regresión Lineal como
 * modelo en c++
 * Requerimientos:
 * - Construir una clase Extraction, que permita
 * manipular, extraer y cargar los datos
 * - Construir una clase LinearRegression, que permita
 * los calculos de la función de costo, gradiente descendiente
 * entre otras.
 */

#include "extractiondata.h"
#include "linearregression.h"
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <list>
#include <vector>
#include <fstream>
#include <boost/algorithm/string.hpp>

int main(int argc, char* argv[])
{
    /*Se crea un objeto del tipo ClassExtraction*/
    ExtractionData ExData(argv[1],argv[2],argv[3]);

    /*Se instancia la clase de Regresión lineal en un objeto*/
    LinearRegression modeloLR;

    /*Se crea un vector de vectores del tipo string para Cargar objeto Exdata lectura*/
    std::vector<std::vector<std::string>> dataframe = ExData.LeerCSV();

    /*Cantidad de filas y columnas*/
    int filas    = dataframe.size();
    int columnas = dataframe[0].size();
    /*Se crea una matriz Eigen, para ingresar los valores a esa matriz*/
    Eigen::MatrixXd matData = ExData.CSVtoEigen(dataframe, filas, columnas);

    /*Se normaliza la matriz de datos*/
    Eigen::MatrixXd mat_Norm = ExData.Normalizar(matData);

    /*Se divide en datos de entrenamiento y datos de prueba*/

    Eigen::MatrixXd X_train, y_train, X_test, y_test;

   //std::cout<<matData<<std::endl;

    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> div_datos = ExData.TrainTestSplit(mat_Norm, 0.8);

    /*Se descomprime la tupla en 4 conjuntos*/
    std::tie(X_train,y_train,X_test,y_test) = div_datos;

    /*Se crea vectores auxiliares para prueba y entrenamiento inicializados en 1*/
    Eigen::VectorXd V_train = Eigen::VectorXd::Ones(X_train.rows());
    Eigen::VectorXd V_test = Eigen::VectorXd::Ones(X_test.rows());
    /*Se redimensiona la matriz de entrenamiento y de prueba para ser ajustadas a
    * los vectores auxiliares anteriores*/
    X_train.conservativeResize(X_train.rows(),X_train.cols()+1);
    X_train.col(X_train.cols()-1) = V_train;
    X_test.conservativeResize(X_test.rows(),X_test.cols()+1);
    X_test.col(X_test.cols()-1) = V_test;

    //std::cout<<X_test<<std::endl;
    /*Se creo el vector de coeficiente theta*/

    Eigen::VectorXd theta = Eigen::VectorXd::Zero(X_test.cols());
    /*Se establece el alpha como ratio de aprendizaje de tipo flotante */
    float alpha = 0.01;
    int num_iter = 1000;



    /*Se crea un vector para almacenar las thetas de salida (parametros m y b)*/
    Eigen::VectorXd thetas_out;
    /*Se crea un vector sencillo (std) de flotantes para almacenar los valores del costo*/
    std::vector<float> costo;
    //Se calcula el gradiente descendiente


    std::tuple<Eigen::VectorXd, std::vector<float>> g_descendiente = modeloLR.GradientDescent(X_test,y_test,
                                                                     theta,
                                                                     alpha,
                                                                     num_iter);
    /*Se desempaqueta el gradiente*/
    std::tie(thetas_out,costo) = g_descendiente; //tie es para descomprimir la tupla


    /*Se almacena los valores de thetas y costos en un fichero para posteriormente ser visualizados*/
    //ExData.VectortoFile(costo,"costos.txt");
    //ExData.EigentoFile(thetas_out, "thetas.txt");

    /*Se extrae el promedio de la matriz de entrada*/
    auto prom_data = ExData.Promedio(matData);

    /*Se extraen los valores de las variables independientes*/
    auto var_prom_independientes = prom_data(0,3);

    /*Se escalan los datos*/
    auto datos_escalados = matData.rowwise()-matData.colwise().mean();

    /*Se extrae la desviación estandar de los datos escalados*/
    auto desv_stand = ExData.DevStand(datos_escalados);

    /*Se extrae los valores de la variable independiente*/
    auto var_desv_independientes = desv_stand(0,3);

    /*Se crea una matriz para almacenar los valores estimados de entrenamiento*/
    Eigen::MatrixXd y_train_hat = (X_train * thetas_out * var_desv_independientes).array() + var_prom_independientes;
    Eigen::MatrixXd x_test_hat = (X_test*thetas_out*var_desv_independientes).array()+var_prom_independientes;

    /*Matriz para los valores reales de y*/
    Eigen::MatrixXd y = matData.col(3).topRows(160);
    Eigen::MatrixXd y1 = matData.col(3).bottomRows(40);
   // std::cout<<y<<std::endl;

    //std::cout<<prom_data<<std::endl;

    /*Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento*/
    float metrica_R2 = modeloLR.R2_Score(y,y_train_hat);
    std::cout<<"Metrica R2 Train: "<<metrica_R2<<std::endl;

    /*Se revisa que tan bueno fue el modelo a traves de la metrica de rendimiento*/
    float metrica_R2_Test = modeloLR.R2_Score(y1,x_test_hat);
    std::cout<<"Metrica R2 Test: "<<metrica_R2_Test<<std::endl;







    return EXIT_SUCCESS;
}




































