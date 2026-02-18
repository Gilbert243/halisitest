import json
import time

class HairClassifier:
    """Hair texture classification using Vision-Language Models.
    
    This classifier sends hair images to a VLM API (e.g., GPT-4V, Gemini)
    with a carefully crafted prompt to classify hair types according to
    the Andre Walker hair typing system (1A-4C).
    
    Features:
    - Automatic retry logic for failed requests
    - JSON response parsing and validation
    - Fallback to 'Unknown' on errors
    - Configurable timeout and temperature
    """
    
    def __init__(self, client, config, prompt: str):
        """Initialize the hair classifier.
        
        Args:
            client: API client (e.g., AzureLLMClient) for making VLM requests
            config: EvalConfig object with API and evaluation settings
            prompt: System prompt text for the VLM
        """
        self.client = client
        self.cfg = config
        self.prompt = prompt
        # Set of allowed hair types for validation
        self.allowed = set(config.evaluation["allowed_types"])

    def _safe_parse(self, raw: str) -> str:
        """Parse and validate JSON response from VLM.
        
        Extracts 'hair_type' field from JSON response and validates
        it against the allowed hair types list.
        
        Args:
            raw: Raw JSON string from VLM API
            
        Returns:
            Validated hair type string or 'Unknown' if invalid
        """
        try:
            # Parse JSON response
            value = json.loads(raw).get("hair_type", "Unknown")
            # Validate against allowed types
            return value if value in self.allowed else "Unknown"
        except Exception:
            # Return Unknown on any parsing error
            return "Unknown"

    def predict(self, image_b64: str) -> str:
        """Classify hair type from base64-encoded image.
        
        Implements retry logic to handle transient API failures.
        Waits 1 second between retries.
        
        Args:
            image_b64: Base64-encoded JPEG image string
            
        Returns:
            Predicted hair type (1A-4C) or 'Unknown' on failure
        """
        # Retry loop for resilience
        for attempt in range(self.cfg.api["retries"] + 1):
            try:
                # Call VLM API
                raw = self.client.classify(
                    model=self.cfg.api["model"],
                    system_prompt=self.prompt,
                    image_b64=image_b64,
                    temperature=self.cfg.api["temperature"],
                    timeout=self.cfg.api["timeout"]
                )
                return self._safe_parse(raw)
            except Exception:
                # Return Unknown after exhausting retries
                if attempt == self.cfg.api["retries"]:
                    return "Unknown"
                # Wait before retry
                time.sleep(1)
