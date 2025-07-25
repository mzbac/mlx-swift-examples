// Copyright © 2024 Apple Inc.

import Foundation
import Hub
import MLX
import MLXLMCommon
import Tokenizers

/// Creates a function that loads a configuration file and instantiates a model with the proper configuration
private func create<C: Codable, M>(
    _ configurationType: C.Type, _ modelInit: @escaping (C) -> M
) -> (URL) throws -> M {
    { url in
        let configuration = try JSONDecoder().decode(
            C.self, from: Data(contentsOf: url))
        return modelInit(configuration)
    }
}

/// Registry of model type, e.g 'llama', to functions that can instantiate the model from configuration.
///
/// Typically called via ``LLMModelFactory/load(hub:configuration:progressHandler:)``.
public class LLMTypeRegistry: ModelTypeRegistry, @unchecked Sendable {

    /// Shared instance with default model types.
    public static let shared: LLMTypeRegistry = .init(creators: all())

    /// All predefined model types.
    private static func all() -> [String: @Sendable (URL) throws -> any LanguageModel] {
        [
            "mistral": create(LlamaConfiguration.self, LlamaModel.init),
            "llama": create(LlamaConfiguration.self, LlamaModel.init),
            "phi": create(PhiConfiguration.self, PhiModel.init),
            "phi3": create(Phi3Configuration.self, Phi3Model.init),
            "phimoe": create(PhiMoEConfiguration.self, PhiMoEModel.init),
            "gemma": create(GemmaConfiguration.self, GemmaModel.init),
            "gemma2": create(Gemma2Configuration.self, Gemma2Model.init),
            "gemma3_text": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "gemma3": create(Gemma3TextConfiguration.self, Gemma3TextModel.init),
            "qwen2": create(Qwen2Configuration.self, Qwen2Model.init),
            "qwen3": create(Qwen3Configuration.self, Qwen3Model.init),
            "qwen3_moe": create(Qwen3MoEConfiguration.self, Qwen3MoEModel.init),
            "starcoder2": create(Starcoder2Configuration.self, Starcoder2Model.init),
            "cohere": create(CohereConfiguration.self, CohereModel.init),
            "openelm": create(OpenElmConfiguration.self, OpenELMModel.init),
            "internlm2": create(InternLM2Configuration.self, InternLM2Model.init),
            "granite": create(GraniteConfiguration.self, GraniteModel.init),
            "mimo": create(MiMoConfiguration.self, MiMoModel.init),
            "glm4": create(GLM4Configuration.self, GLM4Model.init),
        ]
    }

}

/// Registry of models and any overrides that go with them, e.g. prompt augmentation.
/// If asked for an unknown configuration this will use the model/tokenizer as-is.
///
/// The python tokenizers have a very rich set of implementations and configuration.  The
/// swift-tokenizers code handles a good chunk of that and this is a place to augment that
/// implementation, if needed.
public class LLMRegistry: AbstractModelRegistry, @unchecked Sendable {

    /// Shared instance with default model configurations.
    public static let shared = LLMRegistry(modelConfigurations: all())

    static public let smolLM_135M_4bit = ModelConfiguration(
        id: "mlx-community/SmolLM-135M-Instruct-4bit",
        defaultPrompt: "Tell me about the history of Spain."
    )

    static public let mistralNeMo4bit = ModelConfiguration(
        id: "mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        defaultPrompt: "Explain quaternions."
    )

    static public let mistral7B4bit = ModelConfiguration(
        id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        defaultPrompt: "Describe the Swift language."
    )

    static public let codeLlama13b4bit = ModelConfiguration(
        id: "mlx-community/CodeLlama-13b-Instruct-hf-4bit-MLX",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "func sortArray(_ array: [Int]) -> String { <FILL_ME> }"
    )

    static public let deepSeekR1_7B_4bit = ModelConfiguration(
        id: "mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        defaultPrompt: "Is 9.9 greater or 9.11?"
    )

    static public let phi4bit = ModelConfiguration(
        id: "mlx-community/phi-2-hf-4bit-mlx",
        // https://www.promptingguide.ai/models/phi-2
        defaultPrompt: "Why is the sky blue?"
    )

    static public let phi3_5_4bit = ModelConfiguration(
        id: "mlx-community/Phi-3.5-mini-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    )

    static public let phi3_5MoE = ModelConfiguration(
        id: "mlx-community/Phi-3.5-MoE-instruct-4bit",
        defaultPrompt: "What is the gravity on Mars and the moon?",
        extraEOSTokens: ["<|end|>"]
    ) {
        prompt in
        "<|user|>\n\(prompt)<|end|>\n<|assistant|>\n"
    }

    static public let gemma2bQuantized = ModelConfiguration(
        id: "mlx-community/quantized-gemma-2b-it",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "what is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_9b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-9b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_2_2b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-2-2b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        // https://www.promptingguide.ai/models/gemma
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_3_1b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-1b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_3_4b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-4b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_3_12b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-12b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let gemma_3_27b_it_4bit = ModelConfiguration(
        id: "mlx-community/gemma-3-27b-it-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "What is the difference between lettuce and cabbage?"
    )

    static public let qwen205b4bit = ModelConfiguration(
        id: "mlx-community/Qwen1.5-0.5B-Chat-4bit",
        overrideTokenizer: "PreTrainedTokenizer",
        defaultPrompt: "why is the sky blue?"
    )

    static public let qwen2_5_7b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-7B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen2_5_1_5b = ModelConfiguration(
        id: "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_0_6b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-0.6B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_1_7b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-1.7B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_4b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-4B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3_8b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-8B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let qwen3MoE_30b_a3b_4bit = ModelConfiguration(
        id: "mlx-community/Qwen3-30B-A3B-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let openelm270m4bit = ModelConfiguration(
        id: "mlx-community/OpenELM-270M-Instruct",
        // https://huggingface.co/apple/OpenELM
        defaultPrompt: "Once upon a time there was"
    )

    static public let llama3_1_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_8B_4bit = ModelConfiguration(
        id: "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_1B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-1B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let llama3_2_3B_4bit = ModelConfiguration(
        id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
        defaultPrompt: "What is the difference between a fruit and a vegetable?"
    )

    static public let granite3_3_2b_4bit = ModelConfiguration(
        id: "mlx-community/granite-3.3-2b-instruct-4bit",
        defaultPrompt: ""
    )

    static public let mimo_7b_sft_4bit = ModelConfiguration(
        id: "mlx-community/MiMo-7B-SFT-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    static public let glm4_9b_4bit = ModelConfiguration(
        id: "mlx-community/GLM-4-9B-0414-4bit",
        defaultPrompt: "Why is the sky blue?"
    )

    private static func all() -> [ModelConfiguration] {
        [
            codeLlama13b4bit,
            deepSeekR1_7B_4bit,
            gemma2bQuantized,
            gemma_2_2b_it_4bit,
            gemma_2_9b_it_4bit,
            gemma_3_1b_it_4bit,
            gemma_3_4b_it_4bit,
            gemma_3_12b_it_4bit,
            gemma_3_27b_it_4bit,
            granite3_3_2b_4bit,
            llama3_1_8B_4bit,
            llama3_2_1B_4bit,
            llama3_2_3B_4bit,
            llama3_8B_4bit,
            mistral7B4bit,
            mistralNeMo4bit,
            openelm270m4bit,
            phi3_5MoE,
            phi3_5_4bit,
            phi4bit,
            qwen205b4bit,
            qwen2_5_7b,
            qwen2_5_1_5b,
            qwen3_0_6b_4bit,
            qwen3_1_7b_4bit,
            qwen3_4b_4bit,
            qwen3_8b_4bit,
            qwen3MoE_30b_a3b_4bit,
            smolLM_135M_4bit,
            mimo_7b_sft_4bit,
            glm4_9b_4bit,
        ]
    }

}

@available(*, deprecated, renamed: "LLMRegistry", message: "Please use LLMRegistry directly.")
public typealias ModelRegistry = LLMRegistry

private struct LLMUserInputProcessor: UserInputProcessor {

    let tokenizer: Tokenizer
    let configuration: ModelConfiguration
    let messageGenerator: MessageGenerator

    internal init(
        tokenizer: any Tokenizer, configuration: ModelConfiguration,
        messageGenerator: MessageGenerator
    ) {
        self.tokenizer = tokenizer
        self.configuration = configuration
        self.messageGenerator = messageGenerator
    }

    func prepare(input: UserInput) throws -> LMInput {
        let messages = messageGenerator.generate(from: input)
        do {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: input.additionalContext)
            return LMInput(tokens: MLXArray(promptTokens))
        } catch TokenizerError.missingChatTemplate {
            print(
                "No chat template was included or provided, so converting messages to simple text format. This is not optimal for model performance, so applications should provide a chat template if none is included with the model."
            )
            let prompt =
                messages
                .compactMap { $0["content"] as? String }
                .joined(separator: "\n\n")
            let promptTokens = tokenizer.encode(text: prompt)
            return LMInput(tokens: MLXArray(promptTokens))
        }
    }
}

/// Factory for creating new LLMs.
///
/// Callers can use the `shared` instance or create a new instance if custom configuration
/// is required.
///
/// ```swift
/// let modelContainer = try await LLMModelFactory.shared.loadContainer(
///     configuration: LLMRegistry.llama3_8B_4bit)
/// ```
public class LLMModelFactory: ModelFactory {

    public init(typeRegistry: ModelTypeRegistry, modelRegistry: AbstractModelRegistry) {
        self.typeRegistry = typeRegistry
        self.modelRegistry = modelRegistry
    }

    /// Shared instance with default behavior.
    public static let shared = LLMModelFactory(
        typeRegistry: LLMTypeRegistry.shared, modelRegistry: LLMRegistry.shared)

    /// registry of model type, e.g. configuration value `llama` -> configuration and init methods
    public let typeRegistry: ModelTypeRegistry

    /// registry of model id to configuration, e.g. `mlx-community/Llama-3.2-3B-Instruct-4bit`
    public let modelRegistry: AbstractModelRegistry

    public func _load(
        hub: HubApi, configuration: ModelConfiguration,
        progressHandler: @Sendable @escaping (Progress) -> Void
    ) async throws -> ModelContext {
        // download weights and config
        let modelDirectory = try await downloadModel(
            hub: hub, configuration: configuration, progressHandler: progressHandler)

        // Load the generic config to understand which model and how to load the weights
        let configurationURL = modelDirectory.appending(component: "config.json")

        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder().decode(
                BaseConfiguration.self, from: Data(contentsOf: configurationURL))
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        let model: LanguageModel
        do {
            model = try typeRegistry.createModel(
                configuration: configurationURL, modelType: baseConfig.modelType)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, configuration.name, error)
        }

        // apply the weights to the bare model
        try loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: baseConfig.perLayerQuantization)

        let tokenizer = try await loadTokenizer(configuration: configuration, hub: hub)

        let messageGenerator =
            if let model = model as? LLMModel {
                model.messageGenerator(tokenizer: tokenizer)
            } else {
                DefaultMessageGenerator()
            }

        let processor = LLMUserInputProcessor(
            tokenizer: tokenizer, configuration: configuration,
            messageGenerator: messageGenerator)

        return .init(
            configuration: configuration, model: model, processor: processor, tokenizer: tokenizer)
    }

}
