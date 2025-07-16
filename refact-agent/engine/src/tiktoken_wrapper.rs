use std::path::Path;
use std::sync::Arc;
use tiktoken_rs::CoreBPE;
use tokenizers::{Encoding, PaddingParams, TruncationParams};
use serde::{Deserialize, Serialize};
use ahash::AHashMap;
use crate::custom_error::MapErrToString;

/// Configuration structure for TikToken tokenizers
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TikTokenConfig {
    #[serde(default)]
    pub added_tokens_decoder: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub model_max_length: Option<usize>,
    #[serde(default)]
    pub pat_str: Option<String>,
}

impl Default for TikTokenConfig {
    fn default() -> Self {
        Self {
            added_tokens_decoder: std::collections::HashMap::new(),
            model_max_length: None,
            pat_str: None,
        }
    }
}

/// Wrapper around TikToken to provide HuggingFace-compatible interface
#[derive(Clone)]
pub struct TikTokenWrapper {
    tokenizer: Arc<CoreBPE>,
    config: TikTokenConfig,
    truncation: Option<TruncationParams>,
    padding: Option<PaddingParams>,
}

impl std::fmt::Debug for TikTokenWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TikTokenWrapper")
            .field("config", &self.config)
            .field("truncation", &self.truncation)
            .field("padding", &self.padding)
            .finish()
    }
}

impl TikTokenWrapper {
    /// Determine the appropriate tiktoken tokenizer based on config and model path
    fn determine_tokenizer_from_config(config: &TikTokenConfig, model_path: &Path) -> Result<CoreBPE, String> {
        // Try to determine tokenizer type from config or filename
        // This is a heuristic approach until proper model loading is implemented

        // Check if config has model information
        if let Some(pat_str) = &config.pat_str {
            if pat_str.contains("o200k") {
                return tiktoken_rs::o200k_base()
                    .map_err(|e| format!("Failed to load o200k_base tokenizer: {:?}", e));
            }
        }

        // Check filename for hints
        let filename = model_path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("");

        if filename.contains("o200k") || filename.contains("gpt-4o") {
            tiktoken_rs::o200k_base()
                .map_err(|e| format!("Failed to load o200k_base tokenizer: {:?}", e))
        } else if filename.contains("p50k") {
            tiktoken_rs::p50k_base()
                .map_err(|e| format!("Failed to load p50k_base tokenizer: {:?}", e))
        } else if filename.contains("r50k") || filename.contains("gpt2") {
            tiktoken_rs::r50k_base()
                .map_err(|e| format!("Failed to load r50k_base tokenizer: {:?}", e))
        } else {
            // Default to cl100k_base for ChatGPT models
            tracing::warn!("Could not determine tiktoken model type, defaulting to cl100k_base");
            tiktoken_rs::cl100k_base()
                .map_err(|e| format!("Failed to load cl100k_base tokenizer: {:?}", e))
        }
    }

    /// Load TikToken from directory containing tiktoken.model and tokenizer_config.json
    pub fn from_directory(dir_path: &Path) -> Result<Self, String> {
        let model_path = dir_path.join("tiktoken.model");
        let config_path = dir_path.join("tokenizer_config.json");
        
        if !model_path.exists() {
            return Err(format!("tiktoken.model not found in {}", dir_path.display()));
        }
        
        // Load configuration if exists
        let config = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err_with_prefix("Failed to read tokenizer_config.json:")?;
            serde_json::from_str(&config_str)
                .map_err_with_prefix("Failed to parse tokenizer_config.json:")?
        } else {
            TikTokenConfig::default()
        };
        
        // Load TikToken model
        let _model_bytes = std::fs::read(&model_path)
            .map_err_with_prefix("Failed to read tiktoken.model:")?;

        // TODO: Implement proper model loading from tiktoken.model bytes
        // For now, determine the appropriate tokenizer based on config or filename
        let tokenizer = Self::determine_tokenizer_from_config(&config, &model_path)?;
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            config,
            truncation: None,
            padding: None,
        })
    }
    
    /// Create TikTokenWrapper from an existing CoreBPE tokenizer (for testing)
    pub fn from_tokenizer(tokenizer: CoreBPE) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            config: TikTokenConfig::default(),
            truncation: None,
            padding: None,
        }
    }

    /// Load TikToken from a single .model file
    pub fn from_model_file(model_path: &Path) -> Result<Self, String> {
        if !model_path.exists() {
            return Err(format!("Model file not found: {}", model_path.display()));
        }
        
        // Check for config file in same directory
        let config_path = model_path.with_file_name("tokenizer_config.json");
        let config = if config_path.exists() {
            let config_str = std::fs::read_to_string(&config_path)
                .map_err_with_prefix("Failed to read tokenizer_config.json:")?;
            serde_json::from_str(&config_str)
                .map_err_with_prefix("Failed to parse tokenizer_config.json:")?
        } else {
            TikTokenConfig::default()
        };
        
        // Load model
        let _model_bytes = std::fs::read(&model_path)
            .map_err_with_prefix("Failed to read model file:")?;

        // TODO: Implement proper model loading from tiktoken.model bytes
        // For now, determine the appropriate tokenizer based on config or filename
        let tokenizer = Self::determine_tokenizer_from_config(&config, &model_path)?;
        
        Ok(Self {
            tokenizer: Arc::new(tokenizer),
            config,
            truncation: None,
            padding: None,
        })
    }
    
    /// Encode text to tokens with HuggingFace-compatible Encoding
    pub fn encode_fast(&self, text: &str, add_special: bool) -> Result<Encoding, String> {
        let tokens = self.tokenizer.encode_ordinary(text);
        
        // Apply truncation if configured
        let tokens = if let Some(truncation) = &self.truncation {
            let max_length = truncation.max_length;
            if tokens.len() > max_length {
                tokens[..max_length].to_vec()
            } else {
                tokens
            }
        } else {
            tokens
        };
        
        // Convert to HuggingFace Encoding format
        let ids = tokens;
        let type_ids = vec![0u32; ids.len()];

        // Decode individual tokens to get proper token strings
        let tokens_str: Vec<String> = ids.iter()
            .map(|&id| {
                // Try to decode individual token, fallback to ID representation
                self.tokenizer.decode(vec![id])
                    .unwrap_or_else(|_| format!("token_{}", id))
            })
            .collect();

        let word_ids = (0..ids.len()).map(|i| Some(i as u32)).collect();

        // Calculate proper character offsets by accumulating token lengths
        let mut offsets = Vec::new();
        let mut current_offset = 0;
        for token_str in &tokens_str {
            let token_len = token_str.len();
            offsets.push((current_offset, current_offset + token_len));
            current_offset += token_len;
        }

        let special_tokens_mask = vec![0u32; ids.len()];
        let attention_mask = vec![1u32; ids.len()];
        let overflowing = vec![];
        let sequence_ranges = AHashMap::new();
        
        Ok(Encoding::new(
            ids,
            type_ids,
            tokens_str,
            word_ids,
            offsets,
            special_tokens_mask,
            attention_mask,
            overflowing,
            sequence_ranges,
        ))
    }
    
    /// Set truncation parameters
    pub fn with_truncation(&mut self, truncation: Option<TruncationParams>) {
        self.truncation = truncation;
    }
    
    /// Set padding parameters (not implemented for TikToken)
    pub fn with_padding(&mut self, padding: Option<PaddingParams>) {
        if padding.is_some() {
            tracing::warn!("Padding is not supported for TikToken tokenizers");
        }
        self.padding = padding;
    }
}

/// Check if a path contains TikToken format files
/// Only requires tiktoken.model file - tokenizer_config.json is optional
pub fn is_tiktoken_format(path: &Path) -> bool {
    if path.is_dir() {
        // Only require tiktoken.model file, config is optional
        path.join("tiktoken.model").exists()
    } else if path.is_file() {
        // Check if it's a .model file
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext == "model")
            .unwrap_or(false)
    } else {
        false
    }
}
