import { cameraStruct } from "../rendering";

export const mergeShader: string = /* wgsl */ `
${cameraStruct}

struct VertexOutput {
    @builtin(position) Position : vec4<f32>,
    @location(0) textureCoordinates : vec2<f32>,
};
  
@vertex
fn main_vertex(@builtin(vertex_index) vertexIndex : u32)-> VertexOutput {
    return VertexOutput(
        vec4<f32>(
            f32(vertexIndex / u32(2)) * 4.0 - 1.0,
            f32(vertexIndex % u32(2)) * 4.0 - 1.0,
            0.0, 1.0
        ),
        vec2<f32>(
            f32(vertexIndex / u32(2)) * 2.0,
            1.0 - f32(vertexIndex % u32(2)) * 2.0,
        )
    );
}  
  
struct Globals {
    ambientOcclusionCloseTaps: i32,
    ambientOcclusionFarTaps: i32,
    resetAmbientOcclusion: i32,
};

@group(0) @binding(0) var<uniform> globals: Globals;

@group(1) @binding(0) var colorsOpaque : texture_2d<f32>;
@group(1) @binding(1) var colorsVolume : texture_2d<f32>;
@group(1) @binding(2) var ambientOcclusion : texture_2d<f32>;
@group(1) @binding(3) var ambientOcclusionParameters : texture_2d<f32>;
@group(1) @binding(4) var ambientOcclusion2 : texture_2d<f32>;

@fragment
fn main_fragment(@builtin(position) Position : vec4<f32>,
        @location(0) textureCoordinates : vec2<f32>)
    -> @location(0) vec4<f32> {
    let dimensions = vec2<f32>(textureDimensions(colorsOpaque));
    let coordinates = textureCoordinates * dimensions;

    let colorOpaque = textureLoad(colorsOpaque, vec2<i32>(coordinates), 0).xyz;
    let colorVolume: vec4<f32> = textureLoad(colorsVolume, vec2<i32>(coordinates), 0).rgba;
    // let colorTransparent = textureLoad(gBufferColorsTransparent, vec2<i32>(coordinates), 0).rgba;
    // let worldNormalUnit = textureLoad(gBufferWorldNormals, vec2<i32>(coordinates), 0).xyz;
    let aoParameters = textureLoad(ambientOcclusionParameters, vec2<i32>(coordinates), 0).xy;

    let ao = textureLoad(ambientOcclusion, vec2<i32>(coordinates), 0).x / 64.0;
    let ao2 = textureLoad(ambientOcclusion2, vec2<i32>(coordinates), 0).x / 64.0;

    var aoInverse = max(1.0 - (max(ao, ao2) * 1.2), 0);

    // Turns off SSAO if parameter ao.x == 0.0
    if (aoParameters.x == 0.0) {
        aoInverse = 1.0;
    }

    return vec4<f32>((1.0 - colorVolume.a) * (aoInverse * colorOpaque.rgb) + colorVolume.a * colorVolume.rgb, 1.0); 

    // if (aoParameters.x > 0.0) {
    //     return vec4<f32>(aoInverse * colorOpaque.rgb, 1.0);
    // } else {
    //     return vec4<f32>(colorOpaque.rgb, 1.0); 
    // }      
}
`;