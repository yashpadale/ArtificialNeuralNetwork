



        






        
        

        
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

    fn __init__(inout self, rows: Int, cols:Int, val: Scalar[type]):
        self.data = DTypePointer[type].alloc(rows*cols)
        @parameter
        fn splat_val[simd_width: Int](idx: Int) -> None:
            self.data.store[width=simd_width](idx, self.data.load[width=simd_width](idx).splat(val))
        vectorize[splat_val, simd_width](self.rows*self.cols)

    @always_inline
    fn __init__(inout self, other: Self):
        self.data = DTypePointer[type].alloc(rows*cols)
        memcpy(self.data, other.data, rows*cols)



    @always_inline  
    fn __init__(inout self, elems: Int):
        self.data = DTypePointer[type].alloc(elems)
        var rows_ = 1
        var cols = elems
        memset_zero[type](self.data, rows_*self.cols)

    @always_inline
    fn __init__(inout self, rows: Int, cols:Int, *data: Scalar[type]):
        var data_len = len(data)
        self.data = DTypePointer[type].alloc(data_len)
        for i in range(data_len):
            self.data[i] = data[i]

    @always_inline
    fn __init__(inout self, rows: Int, cols:Int, owned list: List[Scalar[type]]):
        var list_len = len(list)
        self.data = DTypePointer[type].alloc(list_len)
        for i in range(list_len):
            self.data[i] = list[i]

    @always_inline
    fn __init__(inout self, dims: StaticIntTuple[2], vals: List[Scalar[type]]):
        var list_len = len(vals)

        self.data = DTypePointer[type].alloc(list_len)
        for i in range(list_len):
            self.data[i] = vals[i]

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.data = DTypePointer[type].alloc(self.rows*self.cols)
        memcpy(self.data, other.data, self.rows*self.cols)


    fn __dim__(inout self)->Tuple[Int,Int]:
        return rows,cols


    # Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        var mean=Float64(-0.1)
        var std=Float64(0.1)
        rand(data, rows * cols)
        #     rand[dtype](res.data(), res.num_elements())

        return Self(data)

    @staticmethod
    fn randn(self) -> Self:
        var mean=Float64(0.01)
        var std=Float64( 0.9)

        var data = DTypePointer[type].alloc(rows * cols)
        randn(data, rows * cols,mean,std**2)
        for i in range(self.rows):
            for j in range(self.cols):
                self[i,j]=self[i,j]*0.01
        return Self(data)

    @staticmethod
    fn randn() -> Self:
        var mean=Float64(-0.01)
        var std=Float64( 0.9)

        var data = DTypePointer[type].alloc(rows * cols)
        randn(data, rows * cols,mean,std**2)
       


        return Self(data)
  

    
   

    fn __setitem__(self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.load[width=nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.store[width=nelts](y * self.cols + x, val)

    @always_inline
    fn __len__(self) -> Int:
        return self.rows*self.cols
    fn __del__(owned self):
        self.data.free()

    @always_inline
    fn __getitem__(self, idx: Int) -> SIMD[type, 1]:
        return self.data.load(idx) 

    @always_inline
    fn __getitem__(self, x: Int, y: Int) -> SIMD[type,1]:
        return self.data.load(x * self.cols + y)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, col: Int) -> Self:
        return self.__getitem__(row_slice, slice(col,col+1))

    @always_inline
    fn __getitem__(self, row: Int, owned col_slice: Slice) -> Self:
        return self.__getitem__(slice(row,row+1),col_slice)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
        self._adjust_row_slice_(row_slice)
        self._adjust_col_slice_(col_slice)

        var src_ptr = self.data
        var dest_mat = Self(row_slice.__len__(),col_slice.__len__())


        for idx_rows in range(row_slice.__len__()):
            src_ptr = self.data.offset(row_slice[idx_rows]*self.cols+col_slice[0])
            @parameter
            fn slice_col_vectorize[simd_width: Int](idx: Int) -> None:
                dest_mat.data.store[width=simd_width](idx+idx_rows*col_slice.__len__(),src_ptr.simd_strided_load[width=simd_width](col_slice.step))
                src_ptr = src_ptr.offset(simd_width*col_slice.step)
            vectorize[slice_col_vectorize, simd_width](col_slice.__len__())
        return dest_mat

    @always_inline
    fn _adjust_row_slice_(self, inout span: Slice):
        if span.start < 0:
            span.start = self.rows + span.start
            
        if not span._has_end():
            span.end = self.rows
        elif span.end < 0:
            span.end = self.rows+ span.end
        if span.end > self.rows:
            span.end = self.rows

    fn _adjust_col_slice_(self, inout span: Slice):
        if span.start < 0:
            span.start = self.cols + span.start
        if not span._has_end():
            span.end = self.cols
        elif span.end < 0:
            span.end = self.cols + span.end
        if span.end > self.cols:
            span.end = self.cols

   

    
  
    
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




       




        




fn MatrixMultiply(A: Matrix, B: Matrix)->Matrix[A.rows,B.cols]:
    if A.cols==B.rows:
        var C=Matrix[A.rows,B.cols]()
        for m in range(C.rows):
            for k in range(A.cols):
                for n in range(C.cols):
                    C[m, n] += A[m, k] * B[k, n]
        return C
    else:
        var empty=Matrix[A.rows,B.cols]()
        print("Shape Mismatch,",A.cols,"!=",B.rows)
        return empty

fn MatrixADD(A: Matrix, B: Matrix)->Matrix[A.rows,A.cols]:
    if A.cols==B.cols:
        var C=Matrix[A.rows,A.cols]()
        for m in range(C.rows):
                for n in range(C.cols):
                    C[m, n] += A[m, n] + B[0, n]
        return C
    else:
        var empty=Matrix[A.rows,A.cols]()
        print("Shape Mismatch")
        return empty


    


fn Forward_Pass(Input:Matrix[],Weights:Matrix[],Bias:Matrix[])->Matrix[Input.rows,Weights.cols]:
    var output=MatrixMultiply(Input,Weights)
    var output1=MatrixADD(output,Bias)
    return output1


fn MeanSquareErrorElementWise(Input_Y:Matrix,Predicted:Matrix)->Matrix[Input_Y.rows,Input_Y.cols]:
    var MeanSqaureError_Element_Wise=Matrix[Input_Y.rows,Input_Y.cols]()

    for i in range(Input_Y.rows):
        for j in range(Input_Y.cols):
            MeanSqaureError_Element_Wise[i,j]=(Predicted[i,j]-Input_Y[i,j])*(Predicted[i,j]-Input_Y[i,j])
    return MeanSqaureError_Element_Wise

fn TotalMeanSquareErrorLoss(MeanSquareErrorElementWise:Matrix)->Float32:
    var Loss:Float32=0
    for i in range(MeanSquareErrorElementWise.rows):
        Loss+=MeanSquareErrorElementWise[i,0]
    return Loss


fn Total(MeanSquareErrorElementWise:Matrix)->Float32:
    var Loss:Float32=0
    for i in range(MeanSquareErrorElementWise.rows):
        Loss+=MeanSquareErrorElementWise[i,0]
    return Loss


fn MatrixMultiplyScaler(M:Matrix,M1:Matrix,Num:Float32):
 
    for i in range(M.rows):
        for j in range(M.rows):
            M1[i,j]=M[i,j]*Num




fn gradient(predicted:Matrix,GroundTruth:Matrix)->Matrix[predicted.rows,predicted.cols]:
    var grad=Matrix[predicted.rows,predicted.cols]()

    for i in range(predicted.rows):
        for j in range(predicted.cols):
            grad[i,j]=(2/GroundTruth.rows)*(predicted[i,j]-GroundTruth[i,j])
    return grad


    

  
fn f():
    var input_x = Matrix[3, 5].randn()
   
    var input_y=Matrix[3, 1].randn()
    


    # Forward Pass Layer1
    var weights1=Matrix[5,1].randn() 
    var bias1=Matrix[2, 1].randn()

   
    var output=Forward_Pass(input_x,weights1,bias1)
    print("input_x")
    print(input_x)
    print("input_y")
    print(input_y)

    print("output")
    print(output)




    var MeanSqaureError_Element_Wise_Matrix=MeanSquareErrorElementWise(input_y,output)
    print("MeanSqaureError_Element_Wise_Matrix")

    print(MeanSqaureError_Element_Wise_Matrix)

    var Loss=TotalMeanSquareErrorLoss(MeanSqaureError_Element_Wise_Matrix)
    print("Loss")

    print(Loss)

    var gradient=gradient(output,input_y)
    print("gradient")

    print(gradient)


    
fn reduce(M1:Matrix):
    for i in range(M1.rows):
        for j in range(M1.cols):
            M1[i,j]=M1[i,j]*0.01



        
fn Transpose(M1:Matrix,M2:Matrix):
        for i in range(M1.rows):
            for  j in range(M1.cols):
                #=M1[i,j]
                M2.__setitem__(j,i,M1[i,j])

fn Change_weights(M1:Matrix,M2:Matrix):
    for i in range(M1.rows):
            for  j in range(M1.cols):
                M1[i,j]=M1[i,j]-M2[i,j]

fn Change_bias(M1:Matrix,M2:Float32,M3:Float32):
    for i in range(M1.rows):
            for  j in range(M1.cols):
                M1[i,j]-=M2*M3

fn con(file_path:String,M:Matrix) raises:
    var pd=Python.import_module("pandas")
    var np=Python.import_module("numpy")



    var   df = pd.read_csv(file_path)
    var r=df.shape[0]
    var c=df.shape[1]
    print("Number of rows : ",r, "Number of cols : ",c)

    for column in df.select_dtypes(include=['object']).columns:
        df[column] = pd.factorize(df[column])[0]

    var np_array = df.to_numpy()

    for i in range(np_array.__len__()):
        var sub=np_array[i]
        for j in range(sub.__len__()):
            var o:Float64=sub[j].to_float64()
            M[i,j]=o
            


fn split(Data:Matrix,x:Matrix,y:Matrix):

    for i in range(Data.rows):
        x[0,i]=Data[0,i]
        
    for i in range(Data.rows):
        y[0,i]=Data[1,i]






fn main () raises:

    var d=Matrix[30,2]()
    var x=Matrix[30,1]()
    var y=Matrix[30,1]()

    
    
    var input_x = Matrix[10, 5].rand()
    var input_y = Matrix[10, 1].rand()

  

    
    var weights1 = Matrix[5, 1].rand()
    var bias1 = Matrix[2, 1].rand()
    reduce(weights1)
    reduce(bias1)
    for i in range(5000):

        var output = Forward_Pass(input_x, weights1, bias1)
        var mse_element_wise = MeanSquareErrorElementWise(input_y, output)
        var loss = TotalMeanSquareErrorLoss(mse_element_wise)
        print("Epoch", i, "Total Loss", loss)

        var grad_output = gradient(output, input_y)

        var input_x_T = Matrix[input_x.cols, input_x.rows]()
        Transpose(input_x, input_x_T)
        var grad_weights1 = MatrixMultiply(input_x_T, grad_output)
        var grad_bias1 = Total(grad_output)

        var lr = 0.01
        var grad_weights_scaled = Matrix[grad_weights1.rows, grad_weights1.cols]()
        
        MatrixMultiplyScaler(grad_weights1, grad_weights_scaled, lr)
        Change_weights(weights1, grad_weights_scaled)

        Change_bias(bias1, lr, grad_bias1)

    
    var input_x_1 = Matrix[10, 5].rand()
    var input_y_2 = Matrix[10, 1].rand()
    var output1 = Forward_Pass(input_x, weights1, bias1)
    var mse_element_wise = MeanSquareErrorElementWise(input_y, output1)
    var loss = TotalMeanSquareErrorLoss(mse_element_wise)



    print("Total Loss : ",loss)
    

    con("/home/yash/my_files/Salary_Data (1).csv",d)

    split(d,x,y)
    print("d dim")
    print(d.rows,d.cols)

    print("x dim")
    print(x.rows,x.cols)

    print("y dim")
    print(y.rows,y.cols)
    
    

 
    
   

    



      









        
      
 
   
   

    



        
   


   



    
    


   



    



    
    
    
    









   
    


    
    




   


      
   

  

    


    




    
   

   
  





    





    
    

    





    



   

