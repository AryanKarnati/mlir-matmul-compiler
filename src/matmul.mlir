// Declare a runtime function that prints an unranked memref of f32
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {

    // Create 3 variables called f1, f2, f3 and initialized them to 2.0, 3.0, 0.0 respectivly
    %c1 = arith.constant 2.0 : f32
    %c2 = arith.constant 3.0 : f32
    %c0 = arith.constant 0.0 : f32

    // Allocates 3 matricies in memory of sizes 2x3.3x2.2x2 respectively
    %m1 = memref.alloc() : memref<2x3xf32>
    %m2 = memref.alloc() : memref<3x2xf32>
    %m0 = memref.alloc() : memref<2x2xf32>

    // Fills all matricies with values from previously decalred variables
    linalg.fill ins(%c1 : f32) outs(%m1 : memref<2x3xf32>)
    linalg.fill ins(%c2 : f32) outs(%m2 : memref<3x2xf32>)
    linalg.fill ins(%c0 : f32) outs(%m0 : memref<2x2xf32>)

    // Does a 2D matrix multiplication
    linalg.matmul ins(%m1, %m2 : memref<2x3xf32>, memref<3x2xf32>) outs(%m0 : memref<2x2xf32>)

    // Casting the rank 2 tensor to unranked for printing
    %cast_m0 = memref.cast %m0 : memref<2x2xf32> to memref<*xf32>
    call @printMemrefF32(%cast_m0) : (memref<*xf32>) -> ()

    // Deallocating memory
    memref.dealloc %m1 : memref<2x3xf32>
    memref.dealloc %m2 : memref<3x2xf32>
    memref.dealloc %m0 : memref<2x2xf32>

    return

}