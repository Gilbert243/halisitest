from openai import AzureOpenAI

class AzureLLMClient:
    """Client for Azure OpenAI Vision-Language Models.
    
    Handles communication with Azure OpenAI API for image classification
    tasks using GPT-4V or similar vision-capable models.
    """
    
    def __init__(self, api_key, endpoint, api_version):
        """Initialize Azure OpenAI client.
        
        Args:
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            api_version: API version (e.g., '2024-02-15-preview')
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version
        )

    def classify(self, model, system_prompt, image_b64, temperature, timeout):
        """Send image classification request to Azure OpenAI.
        
        Args:
            model: Model deployment name (e.g., 'gpt-4o-mini')
            system_prompt: System prompt with classification instructions
            image_b64: Base64-encoded image data
            temperature: Sampling temperature (0 for deterministic)
            timeout: Request timeout in seconds
            
        Returns:
            JSON string with classification result
            
        Raises:
            Exception: On API errors (network, timeout, rate limit, etc.)
        """
        # Create chat completion with vision
        response = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            # Force JSON output format
            response_format={"type": "json_object"},
            messages=[
                # System prompt with task instructions
                {"role": "system", "content": system_prompt},
                # User message with image
                {
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            # Embed image as base64 data URI
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }]
                }
            ],
            timeout=timeout
        )
        # Extract text content from response
        return response.choices[0].message.content
