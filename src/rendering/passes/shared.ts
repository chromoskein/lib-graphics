export interface PassRenderSettings {
    encoder: GPURenderPassEncoder | GPUComputePassEncoder | GPUCommandEncoder | null,
    frameID: number,
}

export abstract class Pass {
    // Ensure that a custom clear color can be set as parameter
    public clearColor: GPUColorDict = { r: 0.0, g: 0.0, b: 0.0, a: 1.0 };

    // Called before the frame before command encoding begins
    public beforeRender(): void {}

    // Called during rendering
    // Either needs a compute or render pass encoders
    // If this render pass middleware actually needs multiple passes then the argument type is of GPUCommandEncoder
    public render(settings: PassRenderSettings): void {}

    // Called at the end of the frame after all the commands buffers generated are submitted to a queue
    public afterRender(): void {}

    //
    public onResize(width: number, height: number): void {}

    public dirty(): boolean { return false; }
}

export const cameraStruct: string = /* wgsl */`
    struct Camera {
        projection: mat4x4<f32>,
        projectionInverse: mat4x4<f32>,
        view: mat4x4<f32>,
        viewInverse: mat4x4<f32>,
        projectionView: mat4x4<f32>,
        projectionViewInverse: mat4x4<f32>,
        normalMatrix: mat4x4<f32>,
        position: vec4<f32>,
        viewportSize: vec2<f32>,
    };
`;

export const rayTracingStructs: string = /* wgsl */`
    struct Ray {
        origin: vec3<f32>,
        direction: vec3<f32>,
    };

    struct Intersection {
        t: f32,
        position: vec3<f32>,
        normal: vec3<f32>,
    };
`;

export const sphereToBoundingRectangleFunction: string = /* wgsl */`
    const D: mat4x4<f32> = mat4x4<f32>(
        vec4<f32>(1.0, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 1.0, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, 1.0, 0.0),
        vec4<f32>(0.0, 0.0, 0.0, -1.0)
    );

    // Find quadratic roots
    fn quadraticRoots(a: f32, b: f32, c: f32) -> vec2<f32> {
        return vec2<f32>(
        ( -b - sqrt( b * b - 4.0 * a * c ) ) / ( 2.0 * a ),
        ( -b + sqrt( b * b - 4.0 * a * c ) ) / ( 2.0 * a )
        );
    }

    fn fullQuad(vertex: u32) -> vec2<f32> {
        var corner: vec2<f32>;
        switch(vertex) {
          case 0: {
            corner = vec2<f32>(-1, 1);
          }
          case 1: {
            corner = vec2<f32>(1, 1);
          }
          case 2: {
            corner = vec2<f32>(-1, -1);
          }
          default: { // 3
            corner = vec2<f32>(1, -1);
          }
        }
    
        return corner;
    }

    fn sphereToBoundingRectangleVertex(position: vec3<f32>, radius: f32, vertex: u32) -> vec2<f32> {
        let T: mat4x4<f32> = mat4x4<f32>(
            vec4<f32>(radius, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, radius, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, radius, 0.0),
            vec4<f32>(position.x, position.y, position.z, 1.0)
        );
    
        let R: mat4x4<f32> = transpose(camera.projectionView * T);
    
        let roots_horizontal: vec2<f32> = quadraticRoots(dot(R[3], D * R[3]), -2.0 * dot(R[0], D * R[3]), dot(R[0], D * R[0]));
        let half_width: f32 = abs(roots_horizontal.x - roots_horizontal.y) * 0.5;
    
        let roots_vertical: vec2<f32> = quadraticRoots(dot(R[3], D * R[3]), -2.0 * dot(R[1], D * R[3]), dot(R[1], D * R[1]));
        let half_height: f32 = abs(roots_vertical.x - roots_vertical.y) * 0.5;
    
        var center: vec4<f32> = vec4<f32>(dot(R[0], D * R[3]), dot(R[1], D * R[3]), 0.0, dot(R[3], D * R[3]));
        center.x = center.x / center.w;
        center.y = center.y / center.w;

        let half_size = vec2<f32>(half_width, half_height);

        var corner: vec2<f32>;
        switch(vertex) {
          case 0: {
            corner = center.xy + vec2<f32>(-half_size.x, half_size.y);
          }
          case 1: {
            corner = center.xy +  vec2<f32>(half_size.x, half_size.y);
          }
          case 2: {
            corner = center.xy + vec2<f32>(-half_size.x, -half_size.y);
          }
          default: { // 3
            corner = center.xy + vec2<f32>(half_size.x, -half_size.y);
          }
        }
    
        return corner;
    }
`;

export const fragmentToRayFunction: string = /* wgsl */`
    fn fragmentSpaceToRay(position: vec4<f32>) -> Ray {
      // Fragment in framebuffer/window coordinates
      var fragmentNormalizedSpace: vec4<f32> = vec4<f32>(position.xyz, 1.0); 
    
      // Fragment in NDC coordinates
      fragmentNormalizedSpace.x = (fragmentNormalizedSpace.x / camera.viewportSize.x) * 2.0 - 1.0;
      fragmentNormalizedSpace.y = (1.0 - (fragmentNormalizedSpace.y / camera.viewportSize.y)) * 2.0 - 1.0;
    
      // Fragment in view space
      var fragmentViewSpace: vec4<f32> = camera.projectionInverse * fragmentNormalizedSpace;
      fragmentViewSpace.z = -1.0;
      fragmentViewSpace.w = 1.0;
    
      // Fragment in world space
      let fragmentWorldSpace: vec4<f32>  = camera.viewInverse * fragmentViewSpace;
    
      // Ray
      return Ray(
        camera.position.xyz,
        normalize((fragmentWorldSpace - camera.position).xyz)
      );
    }
`;

export const phongFunction: string = /* wgsl */`
    // simplified Phong shading model
    fn phong(ray: Ray, intersection: Intersection, color: vec4<f32>) -> vec4<f32> {
        var ambient = 0.05;
        
        var norm =  -intersection.normal;
        var lightDir = ray.direction;
        var diffuse = max(dot(norm, lightDir), 0.0);

        var result = max(ambient + diffuse * 0.5, 1.0) * color.rgb;

        return vec4(result, color.a);
    }
`;
