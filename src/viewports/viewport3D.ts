import { type Camera3D, GraphicsLibrary } from "..";
import { DeferredPass } from "../rendering";
import { Scene } from "../scene";
import { mergeShader } from "./mergeShader";
import { Viewport } from "./viewport";
import * as r from "restructure";
import { ScreenSpaceAmbientOcclusionPass } from "../rendering/passes/ScreenSpaceAmbientOcclusion";

export class ViewportPipelines {
    protected static _instance: ViewportPipelines;
    protected static _lastDeviceUsed: GPUDevice;

    public shaderModules: Map<string, GPUShaderModule>;
    public bindGroupLayouts: Map<string, GPUBindGroupLayout>;
    public pipelineLayouts: Map<string, GPUPipelineLayout>;
    public renderPipelines: Map<string, GPURenderPipeline>;

    private constructor(graphicsLibrary: GraphicsLibrary) {
        const device = graphicsLibrary.device;

        this.shaderModules = new Map();
        this.bindGroupLayouts = new Map();
        this.pipelineLayouts = new Map();
        this.renderPipelines = new Map();

        //#region Graphics Library Requirements
        const cameraBGL = graphicsLibrary.bindGroupLayouts.get("camera");

        if (!cameraBGL) {
            throw "Camera BGL of Graphics Library should have been initialized at this point."
        }
        //#endregion Graphics Library Requirements

        const shaderModule = device.createShaderModule({
            label: "Merge",
            code: mergeShader
        });
        this.shaderModules.set("Merge", shaderModule);

        const globalsBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
            ]
        });
        this.bindGroupLayouts.set("MergeGlobals", globalsBGL);

        const inputTexturesBGL = device.createBindGroupLayout({
            entries: [
                { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
                { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
                { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
                { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
            ]
        });
        this.bindGroupLayouts.set("MergeInputTextures", inputTexturesBGL);

        const pipelineLayout = device.createPipelineLayout({
            bindGroupLayouts: [globalsBGL, inputTexturesBGL],
        });
        this.pipelineLayouts.set("Merge", pipelineLayout);

        this.renderPipelines.set("Merge", device.createRenderPipeline({
            label: "Merge",
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: "main_vertex",
            },
            fragment: {
                module: shaderModule,
                entryPoint: "main_fragment",
                targets: [
                    { format: navigator.gpu.getPreferredCanvasFormat() }
                ]

            },
            primitive: {
                topology: "triangle-strip",
                stripIndexFormat: "uint32",
            },
        }));

        ViewportPipelines._lastDeviceUsed = graphicsLibrary.device;
    }

    public static getInstance(graphicsLibrary: GraphicsLibrary): ViewportPipelines {
        if (this._instance && ViewportPipelines._lastDeviceUsed === graphicsLibrary.device) {
            return this._instance;
        }

        return this._instance = new this(graphicsLibrary);
    }
}

interface MergeShaderGlobals {
    ambientOcclusionCloseTaps: number,
    ambientOcclusionFarTaps: number,
    resetAmbientOcclusion: number,
}

const MergeShaderGlobalsStruct = new r.Struct({
    ambientOcclusionCloseTaps: r.int32le,
    ambientOcclusionFarTaps: r.int32le,
    resetAmbientOcclusion: r.int32le,
});

export class Viewport3D extends Viewport {
    protected _pipelines: ViewportPipelines;

    protected mergeGlobals: MergeShaderGlobals;
    protected mergeGlobalsBuffer: GPUBuffer;
    protected mergeBindGroupGlobals: GPUBindGroup;
    protected mergeBindGroupInputTextures: GPUBindGroup | null = null;

    protected deferredPass: DeferredPass = new DeferredPass(this.graphicsLibrary);
    protected ssaoPass: ScreenSpaceAmbientOcclusionPass = new ScreenSpaceAmbientOcclusionPass(this.graphicsLibrary, 0.3);
    protected ssaoPass2: ScreenSpaceAmbientOcclusionPass = new ScreenSpaceAmbientOcclusionPass(this.graphicsLibrary, 1.0);

    public ssaoEnabled: boolean;

    constructor(graphicsLibrary: GraphicsLibrary, scene: Scene | null = null, camera: Camera3D | null = null, ssaoEnabled: boolean = true) {
        super(graphicsLibrary, scene, camera);
        this.ssaoEnabled = ssaoEnabled;
        this._pipelines = ViewportPipelines.getInstance(graphicsLibrary);


        this.mergeGlobalsBuffer = graphicsLibrary.device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.mergeGlobals = MergeShaderGlobalsStruct.fromBuffer(new Uint8Array(128));
        this.mergeBindGroupGlobals = graphicsLibrary.device.createBindGroup({
            layout: this._pipelines.bindGroupLayouts.get("MergeGlobals")!,
            entries: [{ binding: 0, resource: { buffer: this.mergeGlobalsBuffer } }],
        });
    }

    public setClearColor(color: GPUColorDict): void {
        super.setClearColor(color);
        this.deferredPass.clearColor = color;
    }

    public resize(width: number, height: number): void {
        super.resize(width, height);

        if (width <= 0 || height <= 0) {
            return;
        }

        this.deferredPass.onResize(width, height);
        this.ssaoPass.onResize(width, height);

        this.ssaoPass.depthTextureView = this.deferredPass.depthTexture?.createView() || null;
        this.ssaoPass.normalsTextureView = this.deferredPass.normalsTexture?.createView() || null;

        this.ssaoPass2.onResize(width, height);

        this.ssaoPass2.depthTextureView = this.deferredPass.depthTexture?.createView() || null;
        this.ssaoPass2.normalsTextureView = this.deferredPass.normalsTexture?.createView() || null;
    }

    async render(textureView: GPUTextureView, frametime: number): Promise<void> {
        await super.render(textureView, frametime);

        //#region Preparation
        const device = this.graphicsLibrary.device;

        //#region Camera
        const cameraBindGroupLayout = this.graphicsLibrary.bindGroupLayouts.get("camera");

        if (!this._camera || !this._scene || !cameraBindGroupLayout) {
            return;
        }

        const cameraBindGroup = device.createBindGroup({
            layout: cameraBindGroupLayout,
            entries: [{ binding: 0, resource: { buffer: this._camera.bufferGPU } }]
        });
        //#endregion Camera

        //#region Before Render 
        if (this.dirty) {
            this.ssaoPass.setDirty();
            this.ssaoPass2.setDirty();
        }

        if (this.ssaoEnabled) {
            this.ssaoPass.beforeRender();
            this.ssaoPass2.beforeRender();
        }
        //#endregion Before Render

        //#region Merge Pipeline
        const mergePipeline = this._pipelines.renderPipelines.get("Merge");

        const mergeGlobalsArrayBuffer = new Uint8Array(128);
        mergeGlobalsArrayBuffer.set(MergeShaderGlobalsStruct.toBuffer(this.mergeGlobals), 0);

        device.queue.writeBuffer(this.mergeGlobalsBuffer, 0, mergeGlobalsArrayBuffer, 0, 128);
        //#endregion Merge Pipeline
        //#endregion Preparation

        //#region Render
        const commandEncoder = device.createCommandEncoder();

        this.deferredPass.render({ encoder: commandEncoder, cameraBindGroup, scene: this._scene, frameID: this._frameID });

        if (this.ssaoEnabled){
            const computePassEncoder = commandEncoder.beginComputePass();
            this.ssaoPass.render({ encoder: computePassEncoder, cameraBindGroup, frameID: this._frameID });
            this.ssaoPass2.render({ encoder: computePassEncoder, cameraBindGroup, frameID: this._frameID });
            computePassEncoder.end();
        }
        else{
            // todo: handle ssao disabling - now disabled ssao works only if it was disabled in the constructor
        }

        let mergeBindGroupInputTextures = null;
        const layout = this._pipelines.bindGroupLayouts.get("MergeInputTextures");
        if (this.deferredPass.colorTexture && this.deferredPass.volumeTexture && this.deferredPass.aoTexture && this.ssaoPass.ssaoTexture && this.ssaoPass2.ssaoTexture && layout) {
            mergeBindGroupInputTextures = this.graphicsLibrary.device.createBindGroup({
                layout,
                entries: [
                    { binding: 0, resource: this.deferredPass.colorTexture.createView() },
                    { binding: 1, resource: this.deferredPass.volumeTexture.createView() },
                    { binding: 2, resource: this.ssaoPass.ssaoTexture.createView() },
                    { binding: 3, resource: this.deferredPass.aoTexture.createView() },
                    { binding: 4, resource: this.ssaoPass2.ssaoTexture.createView() },
                ],
            });
        }

        if (this.deferredPass.colorTexture && mergePipeline && mergeBindGroupInputTextures) {
            const mergeRenderPass = commandEncoder.beginRenderPass({
                colorAttachments: [
                    {
                        view: textureView,
                        clearValue: this.clearColor,
                        loadOp: "clear",
                        storeOp: "store",
                    },
                ]
            });
            mergeRenderPass.setPipeline(mergePipeline);
            mergeRenderPass.setBindGroup(0, this.mergeBindGroupGlobals);
            mergeRenderPass.setBindGroup(1, mergeBindGroupInputTextures);
            mergeRenderPass.draw(3, 1, 0, 0);
            mergeRenderPass.end();
        }

        const commandBuffer = commandEncoder.finish();
        //#endregion Render

        // Submit
        device.queue.submit([commandBuffer]);

        this.dirty = false;
    }
}