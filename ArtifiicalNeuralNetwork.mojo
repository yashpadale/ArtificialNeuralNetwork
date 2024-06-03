alias type = DType.float32
from algorithm import vectorize
from memory import memset_zero
from random import rand, random_float64,randn
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from math import mul, div, mod, add, trunc, align_down, align_down_residual,sqrt
from memory import memset_zero, memcpy
from sys.info import simdwidthof
# /home/yash/copyingstuff.mojo
from algorithm import vectorize
from algorithm.functional import elementwise
from algorithm.reduction import max, min, sum, cumsum, mean, argmin
from random import rand
from sys.intrinsics import strided_load
from python import Python
import benchmark
from benchmark import Unit
from testing import assert_true, assert_equal
from buffer import Buffer, NDBuffer
from buffer.list import DimList

from algorithm import parallelize

alias simd_width:Int=simdwidthof[type]()

struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)


    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data


    fn __dim__(inout self)->Tuple[Int,Int]:
        return rows,cols


    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    @staticmethod
    fn randn() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        randn(data, rows * cols)
        return Self(data)

    
    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

 


  
    
    fn __str__(self) -> String:        
        var rank:Int = 2
        var prec:Int = 4
        var printStr:String = ""
        if self.rows == 1:
            rank = 1
        var rows:Int=0
        var cols:Int=0
        if rank==0 or rank>3:
            print("Error: Tensor rank should be: 1,2, or 3. Tensor rank is ", rank)
            return ""
        if rank==1:
            rows = 1
            cols = self.cols
        if rank==2:
            rows = self.rows
            cols = self.cols
        var val:Scalar[type]=0.0
        var ctr: Int = 0
        printStr+=""
        for i in range(rows):
            if rank>1:
                if i==0:
                    printStr+="["
                else:
                    printStr+="\n "
            printStr+="["
            for j in range(cols):  
                if rank==1:
                    val = self.__getitem__(j,1)
                if rank==2:
                    val = self[i,j]
                if type != DType.bool and type != DType.index:
                    var int_str: String
                    if val >= 0.0:
                        int_str = " "+str(trunc(val).cast[DType.index]())
                    else:
                        # val = math.abs(val)
                        int_str = str(trunc(val).cast[DType.index]())
                    var float_str: String = ""
                    if mod(val,1)==0:
                        float_str = "0"
                    else:
                        try:
                            float_str = str(mod(val,1)).split('.')[-1][0:4]
                        except:
                            return ""
                    var s: String = int_str+"."+float_str
                    if j==0:
                        printStr+=s
                    else:
                        printStr+="  "+s
                else:
                    if j==0:
                        printStr+=str(val)
                    else:
                        printStr+="  "+str(val)
            printStr+="]"
        if rank>1:
            printStr+="]"
        printStr+="\n"
        if rank>2:
            printStr+="]"
        printStr+="Matrix: "+str(self.rows)+'x'+str(self.cols)+" | "+"DType:"+str(type)+"\n"
        return printStr

  


alias nelts = simdwidthof[DType.float32]() * 2





    

from algorithm import vectorize


fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):
            @parameter
            fn dot[nelts : Int](n : Int):
                C.store[nelts](m,n, C.load[nelts](m,n) + A[m,k] * B.load[nelts](k,n))
            vectorize[dot, nelts, size = C.cols]()
    parallelize[calc_row](C.rows, C.rows)



fn matadd_parallelized(C: Matrix, A: Matrix):
    for m in range(C.rows):
            for n in range(C.cols):
                C[m, n] += A[0, n]


                



fn mat_subtract_parallelized(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
              
                C[m, n] = (A[m, k] - B[k, n])*(A[m, k] - B[k, n])
               




fn forward_pass(inputs:Matrix[],weights:Matrix[],bias:Matrix[],output:Matrix[])->None:
        matmul_parallelized(output,inputs,weights)
        matadd_parallelized(output,bias)


        


   



    
        

        
       

                
fn main () raises:
    var input_x = Matrix[3, 5].rand()
    var input_y=Matrix[3, 1].rand()

    var weights1=Matrix[5,4].randn() 
    var bias1=Matrix[2, 4].randn()
    var output_layer1=Matrix[3,4]()

    forward_pass(input_x,weights1,bias1,output_layer1)

    var weights2=Matrix[4,1].randn()
    var bias2=Matrix[2, 1].randn()
    var output_layer2=Matrix[3,1]()
    
    forward_pass(output_layer1,weights2,bias2,output_layer2)

    #Mean Square error

    var Loss=Matrix[3,1]()
    mat_subtract_parallelized(Loss,output_layer2,input_y)

    var d=Loss.__dim__()

    var L1=Matrix[1,1]()
    for i in range(d[0]):
      L1[0,0]+=Loss[i,0]


    var MeanSquareError=L1[0,0]
    print(MeanSquareError)


   
    


    
    




   


      
   

  

    


    




    
   

   
  





    





    
    

    





    



   

