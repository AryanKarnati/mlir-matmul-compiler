// Declare a runtime function that prints an unranked memref of f32
func.func private @printMemrefF32(memref<*xf32>)

// creating an identity map
#map_identity_2D = affine_map<(d0, d1) -> (d0, d1)>

func.func @main() {

    // Create 3 variables called f1, f2, f3 and initialized them to 2.0, 3.0, 0.0 respectivly
    %c1 = arith.constant 2.0 : f32
    %c2 = arith.constant 3.0 : f32
    %c3 = arith.constant 5.0 : f32
    %c0 = arith.constant 0.0 : f32


    // Allocates 3 matricies in memory of sizes 2x3.3x2.2x2 respectively
    %m1 = memref.alloc() : memref<512x512xf32>
    %m2 = memref.alloc() : memref<512x512xf32>
    %m3 = memref.alloc() : memref<512x512xf32>
    %m0 = memref.alloc() : memref<512x512xf32>
    %output = memref.alloc() : memref<512x512xf32>
    %relu_out = memref.alloc() : memref<512x512xf32>

    // Fills all matricies with values from previously decalred variables
    linalg.fill ins(%c1 : f32) outs(%m1 : memref<512x512xf32>)
    linalg.fill ins(%c2 : f32) outs(%m2 : memref<512x512xf32>)
    linalg.fill ins(%c3 : f32) outs(%m3 : memref<512x512xf32>)
    linalg.fill ins(%c0 : f32) outs(%m0 : memref<512x512xf32>)
    

    // Does a 2D matrix multiplication
    linalg.matmul ins(%m1, %m2 : memref<512x512xf32>, memref<512x512xf32>) outs(%m0 : memref<512x512xf32>)

    // Adds a bias 
    linalg.add ins(%m0, %m3 : memref<512x512xf32>, memref<512x512xf32>) outs(%output : memref<512x512xf32>)

    // Adds the Relu activation function
    linalg.generic {indexing_maps = [#map_identity_2D, #map_identity_2D], iterator_types = ["parallel", "parallel"]} 
    ins(%output: memref<512x512xf32>) outs(%relu_out : memref<512x512xf32>) 
    {
        ^bb0(%in : f32, %out : f32):
        %zero = arith.constant 0.0 : f32
        %result = arith.maximumf %in, %zero : f32
        linalg.yield %result : f32
    }

    // Casting the rank 2 tensor to unranked for printing
    %cast_m0 = memref.cast %relu_out : memref<512x512xf32> to memref<*xf32>
    call @printMemrefF32(%cast_m0) : (memref<*xf32>) -> ()

    // Deallocating memory
    memref.dealloc %m1 : memref<512x512xf32>
    memref.dealloc %m2 : memref<512x512xf32>
    memref.dealloc %m3 : memref<512x512xf32>
    memref.dealloc %m0 : memref<512x512xf32>
    memref.dealloc %output : memref<512x512xf32>
    memref.dealloc %relu_out : memref<512x512xf32>

    return

}