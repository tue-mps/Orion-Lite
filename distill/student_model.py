import torch
import torch.nn as nn

class OrionStudent(nn.Module):
    def __init__(self, 
                 input_dim=4096, 
                 hidden_dim=1024, 
                 output_dim=4096, 
                 num_layers=6, 
                 num_heads=16, 
                 dropout=0.1):
        super().__init__()
        
        # 1. Input Projection: Compress 4096 -> 1024
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm_input = nn.LayerNorm(hidden_dim)
        
        # 2. Learnable "Planning Query" Token
        # We use this to attend to the visual sequence, similar to how the LLM 
        # generates the next token based on the prompt.
        self.planning_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 3. Transformer Decoder
        # We treat the vision_embeded sequence as "memory" (Key/Value)
        # and our learnable token as the "target" (Query).
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True # Pre-norm is generally more stable
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )
        
        # 4. Output Projection: Expand 1024 -> 4096
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm_output = nn.LayerNorm(output_dim)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vision_embeded):
        """
        Args:
            vision_embeded: (B, N, 4096) - Sequence of visual+map+command tokens
        Returns:
            predicted_states: (B, 1, 4096) - The predicted planning token
        """
        batch_size = vision_embeded.size(0)
        
        # Project Input
        # (B, N, 4096) -> (B, N, 1024)
        memory = self.input_proj(vision_embeded)
        memory = self.norm_input(memory)
        
        # Expand learnable query for batch
        # (1, 1, 1024) -> (B, 1, 1024)
        tgt = self.planning_query.expand(batch_size, -1, -1)
        
        # Pass through Transformer Decoder
        # output shape: (B, 1, 1024)
        # tgt is Query, memory is Key/Value
        out = self.transformer_decoder(tgt, memory)
        
        # Project Output
        # (B, 1, 1024) -> (B, 1, 4096)
        predicted_states = self.output_proj(out)
        predicted_states = self.norm_output(predicted_states)
        
        return predicted_states

if __name__ == "__main__":
    # Test dimensions
    batch_size = 2
    seq_len = 513 # 256 obj + 256 map + 1 command/canbus
    input_dim = 4096
    
    model = OrionStudent()
    dummy_input = torch.randn(batch_size, seq_len, input_dim)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be [2, 1, 4096]