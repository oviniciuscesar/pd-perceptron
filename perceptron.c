



#include "m_pd.h" //importa as funcões prontas do pd
#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "time.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_blas.h"
#include "gsl/gsl_blas_types.h"
#include "gsl/gsl_math.h"



#define MAXRAND 268435457
#define DENRAND 268435456



//define uma nova classe
static t_class *perceptron_class;


//definição da estrutura do objeto - tudo que o objeto precisa (inlets, outlets, variaveis)

typedef struct _perceptron {
    t_object  x_obj;
    t_float   x_neurons; //número de neurônios = número de features de cada dado (mensagem)
    t_float   x_data; //tamanho dos dados de entrada
    t_float   x_wbias; // pesso do bias
    t_float   x_bias;// bias (mensagem)
    t_float   x_class;
    t_float   x_learn; //taxa de aprendizado (mensagem)
    t_float   x_epoca; //número de iterações para treinar o algoritmo (mensagem)
    t_float   x_limiar; // limiar da função degrau (mensagem)
    t_float   x_trainingmode;
    t_float   x_regressionmode;
    t_float   soma;
    t_float   x_exemp; //contador de exemplos
    t_float   x_iter; // contador de iterações
    t_float   x_datasize;
    t_float   x_saida;
    t_float   x_erro;

    
    t_atom    *x_mse;
    t_atom    *x_weights;
    t_atom    *w_bias;


    // t_canvas *x_canvas;
   
    
    
    gsl_vector *data; //nome do vetor para armazenar os dados de entrada
    gsl_vector *pesos; //nome do vetor para armazenar os pesos dos neurônios

    t_outlet  *x_out1; //outlet 1
    t_outlet  *x_out2; //outlet 2
} t_perceptron;




//------------------------ número de neurônios -------------------------//

static void neuron_size (t_perceptron *x, t_floatarg neu){


    if (neu >= 1){
    //desaloca memória
    gsl_vector_free(x->pesos);
    gsl_vector_free(x->data);
    freebytes(x->x_weights, x->x_neurons * sizeof(t_atom));


    //atribui valor à variável x_neurons
    x->x_neurons = neu; 

    //aloca memória novamente
    x->pesos = gsl_vector_alloc (x->x_neurons);
    x->data = gsl_vector_alloc (x->x_neurons); 
    x->x_weights = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    post("neurons: %0.1f", x->x_neurons);
    }
}


//----------------------------- número de épocas -----------------------------//

static void epoch_amount (t_perceptron *x, t_floatarg ep){

    if(ep >=1 ){
        x->x_epoca = ep;
        x->x_erro = 0;
        post("epochs: %0.1f", x->x_epoca);
    }
}

//------------------------- taxa de aprendizado ------------------------------//

static void learning_rate (t_perceptron *x, t_floatarg le){

    if(le >= 0 && le <= 1){
        x->x_learn = le;
        post("learning rate: %0.2f", x->x_learn);
    }
}


//-------------------------- limiar (threshold) ----------------------------//

static void threshold (t_perceptron *x, t_floatarg thr){

    if(thr >= 0 && thr <= 1){
        x->x_limiar = thr;
        post("threshold: %0.2f", x->x_limiar);
    }
}


//------------------------ bias ----------------------------------//

static void bias (t_perceptron *x, t_floatarg bi){
     if(bi >= 0 && bi <= 1){
        x->x_bias = bi;
        post("bias: %0.2f", x->x_bias);
    }
}

static void datasize (t_perceptron *x, t_floatarg dts){
    if (dts >= 1 && dts == trunc(dts)){
        x->x_datasize = (int)dts;
        post("data size: %0.1f", x->x_datasize);    
        }
}


//-------------------- inicialização dos pesos ---------------------------*//

void perceptron_init(t_perceptron *x) {
    int i, j;

    srand(time(NULL));
    

    for (i = 0; i < x->x_neurons; i++){
        gsl_vector_set(x->pesos, i, (double)rand()/RAND_MAX);
        post("weights %0.2f:", gsl_vector_get(x->pesos, i));
    }
    x->x_wbias = (double)rand()/RAND_MAX; //w0 (peso do bias)
    x->x_exemp = 0;
    x->x_iter = 0;
    x->x_erro = 0;
    post ("wbias: %0.3f", x->x_wbias);
}


void zero_init (t_perceptron *x){
    int i;
    gsl_vector_set_zero(x->pesos);
    for (i = 0; i < x->x_neurons; i++){
        post("weights: %0.3f", gsl_vector_get(x->pesos, i));
    }
    x->x_wbias = 0;
    x->x_exemp = 0;
    x->x_iter = 0;
    post ("bias weight: %0.3f", x->x_wbias);
}

static void print (t_perceptron *x) {
    int i;

    for (i = 0; i < x->x_neurons; i++){
        post("weights: %0.3f", gsl_vector_get(x->pesos, i));
        
    }
    
}



//------------ modo de regressão linear --------------//

static void regression (t_perceptron *x, t_floatarg regre){

    x->x_regressionmode = regre;
    int regressionmode = x->x_regressionmode;


    switch(regressionmode) {
        case 0:
            post ("linear regression mode: OFF");
            break;
        case 1: 
            post ("linear regression mode: ON");
            break;
    }

}



static void training (t_perceptron *x, t_floatarg tra){


    x->x_trainingmode = tra;
    int trainingmode = x->x_trainingmode;
    
    switch(trainingmode) {
        case 0:
            post ("training mode: OFF");
            break;
        case 1: 
            post ("training mode: ON");
            break;
    }

}


//--------------------------------- rede treinada ------------------------//

static void trained_mode (t_perceptron *x){

    int i;
    int a;
    float sigmoid;
    float sum = 0;
    int cla = x->x_neurons;


    for (i = 0; i < x->x_neurons; i++) {
            // post("pesos: %0.2f", gsl_vector_get(x->pesos, i));

            sum += gsl_vector_get(x->data, i) * gsl_vector_get(x->pesos, i); //soma da multiplicação dos dados pelos pesos
        }

        x->soma = sum + x->x_wbias * x->x_bias; // soma o peso biais multiplicado pelo bias
        // post("soma: %0.3f", x->soma);
        



    // //*------------------- função de ativação (degrau para classificação e sigmoid para regressão) --------------------*//


    int linear_regression = x->x_regressionmode;
    

    switch (linear_regression){
    case 0:
        
        if (x->soma > x->x_limiar){
            a = 1;
            outlet_float(x->x_out1, a);
        }
        else {
            a = 0;
            outlet_float(x->x_out1, a);
        }

        break;
    case 1:
        
        

        sigmoid = 1.0 / (1.0 + exp(-x->soma));
        outlet_float(x->x_out1, sigmoid);
        // post ("sigmoid: %0.2f", sigmoid);
    }
}






//-------------------------- modo treinamento ------------------------------//

void perceptron_soma (t_perceptron *x, t_symbol *s, int argc, t_atom *argv){

    int i;
    float sum = 0;
    int cla = x->x_neurons;

    if (argc == x->x_neurons+1){

        for (i = 0; i < x->x_neurons; i++) {
            gsl_vector_set(x->data, i, argv[i].a_w.w_float);
            // post("dados: %0.2f", gsl_vector_get(x->data, i));
        }
    
    x->x_class = argv[cla].a_w.w_float; //o ultimo elemento do vetor vai ser a classe do exemplo enviado à rede
    // post("classe: %0.3f", x->x_class);

    int mode = x->x_trainingmode;

    switch (mode){
    case 0:
        trained_mode(x);
        break;
    case 1:

   

    for (i = 0; i < x->x_neurons; i++) {
            // post("pesos: %0.2f", gsl_vector_get(x->pesos, i));
            sum += gsl_vector_get(x->data, i) * gsl_vector_get(x->pesos, i); //soma da multiplicação dos dados pelos pesos
            // post("soma ponderada: %0.3f", sum);
            
            
        
        }

        x->soma = sum + x->x_wbias * x->x_bias; // soma o peso biais multiplicado pelo bias
        // post("multioma+bias: %0.3f", x->soma);

        
    


    // //*------------------- função de ativação (degrau) --------------------*//

        int a;
        float error;
        



        if (x->soma > x->x_limiar){
            a = 1;
            outlet_float(x->x_out1, a);
        }
        else {
            a = 0;
            outlet_float(x->x_out1, a);
    
        if (a != x->x_class){ //se a rede errou incrementa o contador de erros
            x->x_erro++; 
        }

    //  error = (x->x_class - x->soma) * (x->x_class - x->soma);
    //  post ("error: %0.2f", error);
   

        }

// //--------------------  atualização dos pesos  ----------------------------//
        int neurons;

        neurons = x->x_neurons;

        x->x_weights[neurons].a_type = A_FLOAT;
        // x->w_bias[1].a_type = A_FLOAT;

            if (x->x_class != x->soma){

                for (i = 0; i < x->x_neurons; i++){
                    gsl_vector_set(x->pesos, i, gsl_vector_get(x->pesos, i) + x->x_learn * (x->x_class - a) * gsl_vector_get(x->data, i)); // atualiza os pesos
                    // post("updated: %0.3f", gsl_vector_get(x->pesos, i));

                    SETFLOAT (x->x_weights + i, gsl_vector_get(x->pesos, i));
                    
                }

                outlet_anything(x->x_out2, gensym("weights"), neurons, x->x_weights);
                
                x->x_wbias = x->x_wbias + x->x_learn * (x->x_class - a) * x->x_bias; // atualiza o peso do bias
                // post ("bias updated %0.3f", x->x_wbias);
                

        

                SETFLOAT (x->w_bias, x->x_wbias);
                outlet_anything(x->x_out2, gensym("wbias"), 1, x->w_bias);
            }

            x->x_exemp++; //incrementa o contador de exemplos enviados a rede

            if (x->x_exemp == x->x_datasize){  // verifica se o número de exemplos é igual ao tamanho do dataset, se for, incrementa o contador de épocas
                x->x_iter++;
                x->x_exemp = 0; //reinicia o contador de exemplos enviado à rede
                
                
                x->x_mse[1].a_type = A_FLOAT;
                SETFLOAT (x->x_mse, x->x_erro);
                outlet_anything(x->x_out2, gensym("error"), 1, x->x_mse);
                post("epoch: %0.1f", x->x_iter);
                x->x_erro = 0; //reiniciia o contador de erros
             
            }

            if (x->x_iter == x->x_epoca){
                x->x_trainingmode = 0;
                post ("the training process has reached the maximum amount of epochs");
                               
            }

            
        }
     }
}


//--------------------------- salva os pesos em um arquivo de texto --------------------------------//

// static void perceptron_write (t_perceptron *x, t_symbol *filename) {

//    FILE *fd;
//     char buf[MAXPDSTRING];
//     int i;
//     canvas_makefilename(x->x_canvas, filename->s_name,
//         buf, MAXPDSTRING);
//     sys_bashfilename(buf, buf);
//     if (!(fd = fopen(buf, "w")))
//     {
//         error("%s: can't create", buf);
//         return;
//     }
//     for (i = 0; i < x->x_neurons; i++) {
//         if (i > 0) {
//             fprintf(fd, " ");
//         }
//         if (fprintf(fd, "%g", gsl_vector_get(x->pesos, i)) < 0) {
//             error("%s: write error", filename->s_name);
//             goto fail;
//         }
//     }
//     fprintf(fd, "\n");
//     fclose(fd);
//     post("file saved");
//     return;
// fail:
//     fclose(fd);
//     post("file save failed");
// }




//construtor do objeto (aqui também deve ser declarado os inlets, outlets e argumentos do objeto e tudo que tiver que ser executado quando o objeto for criado)

static void *perceptron_new(t_symbol *s, int argc, t_atom *argv) { //argc é a quantidade de elementos da lista e argv é um ponteiro para uma lista
    t_perceptron *x = (t_perceptron *)pd_new(perceptron_class);
    

    x->x_neurons = 2;
    x->x_erro = 0;
    x->x_bias = 1; //
    x->x_learn = 0.5;
    x->x_limiar = 0.5;
    x->x_epoca = 100;
    x->x_iter = 0;
    x->x_exemp = 0;
    x->data = gsl_vector_alloc (x->x_neurons);
    x->pesos = gsl_vector_alloc (x->x_neurons);
    x->x_mse = (t_atom *)getbytes(1 * sizeof(t_atom));
    x->x_weights = (t_atom *)getbytes(x->x_neurons * sizeof(t_atom));
    x->w_bias = (t_atom *)getbytes(1 * sizeof(t_atom));   
    perceptron_init(x);
    // x->x_canvas = canvas_getcurrent();
    x->x_out1 = outlet_new(&x->x_obj, &s_float);
    x->x_out2 = outlet_new(&x->x_obj, &s_anything);
    post("percetron v0.1");
    return (void *)x;
}







//função para destruir o objeto (tudo que precisar alocar memória deve ser destruído aqui)
void perceptron_destroy(t_perceptron *x) { 
	
    gsl_vector_free (x->data);
    gsl_vector_free (x->pesos);
    freebytes(x->x_mse, 1 * sizeof(t_atom));
    freebytes(x->x_weights, x->x_neurons * sizeof(t_atom));
    freebytes(x->w_bias, 1 * sizeof(t_atom));  

	outlet_free(x->x_out1);
    outlet_free(x->x_out2); //desaloca memoria do outlet quando o objeto é destruido
}


    



//inicialização da classe - quando o objeto é carregado pelo pd essa função é ativada
void perceptron_setup(void) {
	perceptron_class = class_new(
		gensym("perceptron"), //nome do objeto
		(t_newmethod)perceptron_new, //chama a função construtor
		(t_method)perceptron_destroy, //chama a função destruidor
		sizeof(t_perceptron),
        CLASS_DEFAULT,
         A_DEFFLOAT, 0);//tamanho do objeto
		 //objeto padrão do pd com inlet talvez a solução era criar objeto sem inlet e adicionar inlet quente 
		//PARAMETROS
	  //último parâmetro é sempre zero para indicar o fim
	class_addlist(perceptron_class, (t_method) perceptron_soma);
    class_addmethod(perceptron_class, (t_method) neuron_size, gensym("weights"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) epoch_amount, gensym("epochs"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) learning_rate, gensym("learning"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) threshold, gensym("threshold"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) bias, gensym("bias"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) perceptron_init, gensym("reset"), A_GIMME, 0);
    class_addmethod(perceptron_class, (t_method) training, gensym("training"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) regression, gensym("regression"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) datasize, gensym("datasize"), A_FLOAT, 0);
    class_addmethod(perceptron_class, (t_method) zero_init, gensym("zero"), A_GIMME, 0);
    // class_addmethod(perceptron_class, (t_method) perceptron_write, gensym("write"), A_SYMBOL, 0);
    class_addmethod(perceptron_class, (t_method) print, gensym("print"), A_GIMME, 0);
}








