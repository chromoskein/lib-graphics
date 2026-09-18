import { GraphicsLibrary, type PassRenderSettings } from "../../..";
import { Pass } from "../shared";
import { BlueNoise } from "./noise";
import { Pipelines } from "./pipelines";
import * as r from "restructure";
import { vec3 } from "gl-matrix";

const lerp = (a: number, b: number, f: number): number => {
    return a + f * (b - a);
};

export interface ScreenSpaceAmbientOcclusionPassSettings extends PassRenderSettings {
    encoder: GPUComputePassEncoder,
    cameraBindGroup: GPUBindGroup,
}

export interface ScreenSpaceAmbientOcclusionGlobals {
    noiseSamples: number[];
    samplesPerPass: number;
    radius: number;
}

export const ScreenSpaceAmbientOcclusionGlobalsStruct = new r.Struct({
    noiseSamples: new r.Array(r.floatle, 1024),
    samplesPerPass: r.int32le,
    radius: r.floatle,
});

export class ScreenSpaceAmbientOcclusionPass extends Pass {
    private width: number = 0;
    private height: number = 0;

    private _graphicsLibrary: GraphicsLibrary;
    private _pipelines: Pipelines;
    private _noise: BlueNoise | null = null;

    private _globals: ScreenSpaceAmbientOcclusionGlobals = ScreenSpaceAmbientOcclusionGlobalsStruct.fromBuffer(new Uint8Array(8192));
    private _globalsBuffer: GPUBuffer;
    private _ssaoTextures: GPUTexture[];

    private currentTexture = 0;
    private _dirty = true;

    private _depthTextureView: GPUTextureView | null = null;
    private _normalsTextureView: GPUTextureView | null = null;

    private globalsBindGroupEven: GPUBindGroup | null = null;
    private globalsBindGroupOdd: GPUBindGroup | null = null;
    private gBufferBindGroup: GPUBindGroup | null = null;

    private _accumulatedSamplesCount = 0;

    private randomScale = 0.62;

    constructor(graphicsLibrary: GraphicsLibrary, radius: number = 0.1, depthTextureView = null, normalsTextureView = null) {
        super();

        this._graphicsLibrary = graphicsLibrary;
        this._pipelines = Pipelines.getInstance(graphicsLibrary);
        this._ssaoTextures = [];

        this._globalsBuffer = graphicsLibrary.device.createBuffer({
            size: 8192,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._globals.radius = radius;
        this._globals.samplesPerPass = 64;

        this._depthTextureView = depthTextureView;
        this._normalsTextureView = normalsTextureView;

        BlueNoise.getInstance(graphicsLibrary).then((noise) => this._noise = noise);
    }

    private createTextures(): void {
        this._ssaoTextures.forEach(t => t.destroy());

        for (let i = 0; i < 2; i++) {
            this._ssaoTextures[i] = this._graphicsLibrary.device.createTexture({
                format: "r32float",
                size: { width: this.width, height: this.height },
                usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.STORAGE_BINDING
            });
        }
    }

    public onResize(width: number, height: number): void {
        this.width = width;
        this.height = height;

        this.createTextures();

        this._dirty = true;
    }

    public beforeRender(): void {
        const globalsBGL = this._pipelines.bindGroupLayouts.get("Globals");
        const inputBGL = this._pipelines.bindGroupLayouts.get("Input");

        if (this._dirty
            && this.width > 0
            && this.height > 0
            && globalsBGL
            && inputBGL
            && this._noise
            && this._depthTextureView && this._normalsTextureView) {
            // Recompute samples
            for (let i = 0; i < 256; i++) {
                const sample = vec3.clone(this._noise.normalizedImageSamples[i]);
                vec3.scale(sample, sample, this.randomScale);

                const scale = lerp(0.1, 1.0, (i / 64.0) * (i / 64.0));
                vec3.scale(sample, sample, scale);

                this._globals.noiseSamples[i * 4] = sample[0];
                this._globals.noiseSamples[i * 4 + 1] = sample[1];
                this._globals.noiseSamples[i * 4 + 2] = sample[2];
            }

            // Write new globals
            const u8View = ScreenSpaceAmbientOcclusionGlobalsStruct.toBuffer(this._globals);
            this._graphicsLibrary.device.queue.writeBuffer(
                this._globalsBuffer,
                0,
                u8View.buffer,
                u8View.byteOffset,
                u8View.byteLength,
            )

            // Clear the SSAO texture
            const nullBuffer = new Uint8Array(4 * (this.width + this.width % 64) * this.height);
            nullBuffer.fill(0.0);
            this._graphicsLibrary.device.queue.writeTexture(
                { texture: this._ssaoTextures[0] },
                nullBuffer,
                { bytesPerRow: (this.width + this.width % 64) * 4, }, { width: this.width, height: this.height }
            );

            // Recreate bind groups
            this.globalsBindGroupEven = this._graphicsLibrary.device.createBindGroup({
                label: "SSAO Globals Even",
                layout: globalsBGL,
                entries: [
                    { binding: 0, resource: { buffer: this._globalsBuffer, offset: 0 } },
                    { binding: 1, resource: this._noise.noiseTexture.createView() },
                    { binding: 2, resource: this._ssaoTextures[0].createView() },
                    { binding: 3, resource: this._ssaoTextures[1].createView() },
                ]
            });

            this.globalsBindGroupOdd = this._graphicsLibrary.device.createBindGroup({
                label: "SSAO Globals Odd",
                layout: globalsBGL,
                entries: [
                    { binding: 0, resource: { buffer: this._globalsBuffer, offset: 0 } },
                    { binding: 1, resource: this._noise.noiseTexture.createView() },
                    { binding: 2, resource: this._ssaoTextures[1].createView() },
                    { binding: 3, resource: this._ssaoTextures[0].createView() },
                ]
            });

            this.gBufferBindGroup = this._graphicsLibrary.device.createBindGroup({
                label: "SSAO G-Buffer",
                layout: inputBGL,
                entries: [
                    { binding: 0, resource: this._depthTextureView },
                    { binding: 1, resource: this._normalsTextureView },
                ]
            });

            this.currentTexture = 0;
            this._accumulatedSamplesCount = 0;
            this._dirty = false;
        }
    }

    public render({ encoder, cameraBindGroup }: ScreenSpaceAmbientOcclusionPassSettings): void {
        const pipeline = this._pipelines.computePipelines.get("SSAO");

        if (this._ssaoTextures.length < 2
            || this._noise === null
            || !this.globalsBindGroupEven || !this.globalsBindGroupOdd || !this.gBufferBindGroup || !pipeline) {
            return;
        }

        // if (this._accumulatedSamplesCount >= 64) {
        //     return;
        // }

        encoder.setPipeline(pipeline);
        encoder.setBindGroup(0, cameraBindGroup);
        encoder.setBindGroup(1, this.currentTexture === 0 ? this.globalsBindGroupEven : this.globalsBindGroupOdd);
        encoder.setBindGroup(2, this.gBufferBindGroup);
        encoder.dispatchWorkgroups(
            Math.ceil((this.width + 7) / 8),
            Math.ceil((this.height + 7) / 8),
        );

        this._accumulatedSamplesCount += this._globals.samplesPerPass;
        // this.currentTexture = (this.currentTexture + 1) % 2;
    }

    public get ssaoTexture(): GPUTexture | null {
        return this._ssaoTextures.length === 2 ? this._ssaoTextures[1] : null;
    }

    public set depthTextureView(depthTextureView: GPUTextureView | null) {
        this._depthTextureView = depthTextureView;
        this._dirty = true;
    }

    public set normalsTextureView(normalsTextureView: GPUTextureView | null) {
        this._normalsTextureView = normalsTextureView;
        this._dirty = true;
    }

    public set radius(radius: number) {
        this._globals.radius = radius;
        this._dirty = true;
    }

    public set samplesPerPass(samplesPerPass: number) {
        this.samplesPerPass = samplesPerPass;
        this._dirty = true;
    }

    public get accumulatedSamplesCount(): number {
        return this._accumulatedSamplesCount;
    }

    public setDirty(): void { this._dirty = true; }
    public dirty(): boolean { return this._dirty; }
}
