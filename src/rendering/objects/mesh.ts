import { type BoundingBox } from "../../shared";
import { ConcreteObject } from "./shared";
import { GraphicsLibrary } from "../..";
import { Allocator } from "../../allocation";
import * as r from "restructure";
import { mat4, vec4, vec3 } from "gl-matrix";

export interface Vertex {
    position: [number, number, number, number];
    normal: [number, number, number, number];
}

export const VertexStruct = new r.Struct({
    position: new r.Array(r.floatle, 4),
    normal: new r.Array(r.floatle, 4),
});

export interface MeshUniform {
    modelMatrix: mat4;
    translate: vec4;
    scale: vec4;
    color: vec4;
}

export const MeshUniformStruct = new r.Struct({
    modelMatrix: new r.Array(r.floatle, 16),
    translate: new r.Array(r.floatle, 4),
    scale: new r.Array(r.floatle, 4),
    color: new r.Array(r.floatle, 4),
});

export class Mesh extends ConcreteObject {
    public static variableName = "mesh";
    public static typeName = "Mesh";

    public getVariableName(): string { return Mesh.variableName }
    public getTypeName(): string { return Mesh.typeName }

    static bindGroupLayouts: Array<GPUBindGroupLayout> = [];
    static createBindGroupLayouts(device: GPUDevice): void {
        Mesh.bindGroupLayouts = [device.createBindGroupLayout({
            label: this.typeName,
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "uniform" },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: "read-only-storage" },
            }]
        })];
    }

    //#region GPU Code
    public static gpuGlobals = /* wgsl */`
        struct MeshUniform {
            modelMatrix: mat4x4<f32>,
            translate: vec4<f32>,
            scale: vec4<f32>,
            color: vec4<f32>,
        };

        struct Vertex {
            position: vec4<f32>,
            normal: vec4<f32>,
        };

        @group(1) @binding(0) var<uniform> meshUniform: MeshUniform;
        @group(1) @binding(1) var<storage, read> vertices: array<Vertex>;
    `;

    public static gpuVertexShader = /* wgsl */`
        struct VertexOutput {
            @builtin(position) Position: vec4<f32>,
            @location(0) @interpolate(flat) instanceIndex: u32,
            @location(1) @interpolate(linear) barycentric: vec3<f32>,
            @location(2) @interpolate(linear) normal: vec4<f32>,
        };
        
        @vertex
        fn main_vertex(
            @builtin(vertex_index) vertexIndex: u32, 
            @builtin(instance_index) instanceIndex: u32) 
        -> VertexOutput {
            var vertex = vertices[vertexIndex];
            var position = vertex.position;
            var normal = vertex.normal;
            var barycentric = vec3<f32>(1.0, 0.0, 0.0);

            if (vertexIndex % 3 == 1) {
                barycentric = vec3<f32>(0.0, 1.0, 0.0);
            }

            if (vertexIndex % 3 == 2) {
                barycentric = vec3<f32>(0.0, 0.0, 1.0);
            }


            var a0 = meshUniform.modelMatrix[0].xyz;
            var a1 = meshUniform.modelMatrix[1].xyz;
            var a2 = meshUniform.modelMatrix[2].xyz;

            var worldNormal = cross(a1, a2) * normal.x + cross(a2, a0) * normal.y + cross(a0, a1) * normal.z;
            if (dot(a0, cross(a1, a2)) < 0.0) {
                worldNormal = -worldNormal;
            }

            var magnitude = length(worldNormal);

            return VertexOutput(
                camera.projectionView * meshUniform.modelMatrix * vec4<f32>(position.xyz, 1.0),
                instanceIndex,
                barycentric,
                vec4<f32>(normal.xyz, 0.0)
            );
        }
    `;

    public static gpuFragmentShader = /* wgsl */`
        //if (vertexOutput.barycentric.x < 0.01 || vertexOutput.barycentric.y < 0.01 || vertexOutput.barycentric.z < 0.01) {
        //if (all(vertexOutput.barycentric > vec3<f32>(0.03))) {
        //    discard;
        //}
        //}
    `;

    static gpuCodeGetOutputValue(variable: "color" | "normal" | "ao"): string {
        switch (variable) {
            case "color": {
                return /* wgsl */`
                    let ray = fragmentSpaceToRay(vertexOutput.Position);
                    var ambient = 0.7;
        
                    var norm =  -vertexOutput.normal.xyz;
                    var lightDir = ray.direction;
                    var diffuse = max(dot(norm, lightDir), 0.0);
                    var phong = min(ambient + diffuse * 0.3, 1.0) * meshUniform.color.rgb;
                    let color = vec4<f32>(phong.rgb, 1.0);
                `;
            }
            case "normal": {
                return /* wgsl */`
                    let normal = normalize(vertexOutput.normal);
                `;
            }
            case "ao": {
                return /* wgsl */`
                    let ao = vec2(1.0, 1.0);
                `;
            }
        }
    }
    //#endregion GPU Code

    public uniform: MeshUniform;

    protected _trianglesCount;
    protected _vertices: Array<Vertex> = [];

    protected _bindGroup: GPUBindGroup | null = null;

    protected trianglesRestructure;

    protected bytes;

    constructor(id: number, graphicsLibrary: GraphicsLibrary, allocator: Allocator, trianglesCount = 1) {
        super(id, graphicsLibrary, allocator);

        this.uniform = {
            modelMatrix: mat4.create(),
            translate: vec4.fromValues(0.0, 0.0, 0.0, 1.0),
            scale: vec4.fromValues(1.0, 1.0, 1.0, 1.0),
            color: vec4.fromValues(1.0, 1.0, 1.0, 1.0),
        };
        this._trianglesCount = trianglesCount;
        this._vertices = new Array(trianglesCount * 3);
        for (let i = 0; i < trianglesCount * 3; i++) {
            this._vertices[i] = {
                position: [0.0, 0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 0.0, 0.0],
            };
        }
        this._allocation = allocator.allocate(512 + trianglesCount * 3 * 32);

        this.trianglesRestructure = new r.Array(VertexStruct, this._vertices.length);

        this.bytes = new Uint8Array(512 + this._vertices.length * 32);

        this.onAllocationMoved();
    }

    public onAllocationMoved(): void {
        if (!this._allocation) {
            return;
        }

        this._bindGroup = this._graphicsLibrary.device.createBindGroup({
            layout: Mesh.bindGroupLayouts[0],
            entries: [
                {
                    binding: 0, resource: {
                        buffer: this._allocation.gpuBuffer.inner,
                        offset: this._allocation.allocationRange.offset,
                        size: 512,
                    }
                },
                {
                    binding: 1, resource: {
                        buffer: this._allocation.gpuBuffer.inner,
                        offset: this._allocation.allocationRange.offset + 512,
                        size: this._trianglesCount * 3 * 32,
                    }
                }
            ]
        });

        this.toBuffer(this._allocation.cpuBuffer.inner, this._allocation.allocationRange.offset);
    }

    public record(encoder: GPURenderPassEncoder, bindGroupLayoutsOffset = 1): void {
        if (!this._bindGroup) {
            return;
        }

        // Set bind group
        encoder.setBindGroup(bindGroupLayoutsOffset + 0, this._bindGroup);

        // Draw
        encoder.draw(this._trianglesCount * 3, 1, 0, 0);
    }

    public toBuffer(buffer: ArrayBuffer, offset: number): void {
        const u8View = new Uint8Array(buffer, offset);

        // u8View.set(MeshUniformStruct.toBuffer(this.uniform), 0);
        // u8View.set(trianglesRestructure.toBuffer(this._vertices), 512);

        u8View.set(this.bytes);
    }

    public toBoundingBoxes(): BoundingBox[] {
        return []
    }

    public set translate(t: vec3) {
        this.uniform.translate = [t[0], t[1], t[2], 1.0];
        this.uniform.modelMatrix = mat4.create();
        mat4.scale(this.uniform.modelMatrix, this.uniform.modelMatrix, vec3.fromValues(this.uniform.scale[0], this.uniform.scale[1], this.uniform.scale[2]));
        mat4.translate(this.uniform.modelMatrix, this.uniform.modelMatrix, t);

        this._dirtyCPU = true;
        this._dirtyGPU = true;

        this.bytes.set(MeshUniformStruct.toBuffer(this.uniform), 0);
    }

    public set scale(s: number) {
        this.uniform.scale = [s, s, s, 1.0];
        this.uniform.modelMatrix = mat4.create();
        mat4.scale(this.uniform.modelMatrix, this.uniform.modelMatrix, vec3.fromValues(s, s, s));
        // mat4.translate(this.uniform.modelMatrix, this.uniform.modelMatrix, vec3.fromValues(this.uniform.translate[0], this.uniform.translate[1], this.uniform.translate[2]));      

        this.uniform.modelMatrix[12] = this.uniform.translate[0];
        this.uniform.modelMatrix[13] = this.uniform.translate[1];
        this.uniform.modelMatrix[14] = this.uniform.translate[2];
        this.uniform.modelMatrix[15] = 1.0;

        this._dirtyCPU = true;
        this._dirtyGPU = true;

        this.bytes.set(MeshUniformStruct.toBuffer(this.uniform), 0);
    }

    public set color(c: vec4) {
        this.uniform.color = c;

        this._dirtyCPU = true;
        this._dirtyGPU = true;

        this.bytes.set(MeshUniformStruct.toBuffer(this.uniform), 0);
    }

    public set vertices(vertices: Array<Vertex>) {
        this._vertices = vertices;

        this.bytes.set(this.trianglesRestructure.toBuffer(this._vertices), 512);
    }

    public set verticesBytes(vertices: Uint8Array) {
        this.bytes.set(vertices, 512);
    }

    public get vertices(): Array<Vertex> {
        return this._vertices;
    }
}