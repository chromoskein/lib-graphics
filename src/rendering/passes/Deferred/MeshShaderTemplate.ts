import { cameraStruct, fragmentToRayFunction, phongFunction, rayTracingStructs } from "../shared";

export function meshShaderTemplate(
    globals: string,
    
    vertexShader: string,
    fragmentCode: string,

    propertyGetters: {
        color: string,
        normal: string,
        ao: string,
    }
): string {
    return /* wgsl */`
${cameraStruct}

@group(0) @binding(0) var<uniform> camera: Camera;

${rayTracingStructs}

${fragmentToRayFunction}
${phongFunction}

${globals}
${vertexShader}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) ao: vec2<f32>,
};

@fragment
fn main_fragment(vertexOutput: VertexOutput) -> FragmentOutput {
    ${fragmentCode}

    // 4. Get values & write them (renderPass asks for and decides!)
    ${propertyGetters.color}
    ${propertyGetters.normal}

    return FragmentOutput(
        vec4<f32>(color.rgb, 1.0),
        vec4<f32>(normal.rgb, 1.0),
        vec2<f32>(1.0, 1.0)
    );
}
`}
