import logging
import os
import openai
from typing import Any, Dict, Tuple

from minions.usage import Usage


class WebSearchClient:
    def __init__(
            self,
            model_name: str = "gpt-4o",  # or "gpt-4o-mini"
            api_key: str = None,
            max_results: int = 3,
            url_included: bool = False,
    ):
        """
        Initialize the OpenAI Web Search Client.

        Args:
            model_name: Model to use (default: "gpt-4o").
            api_key: OpenAI API key (falls back to environment variable if not provided).
            max_results: Maximum number of results to include (default: 3).
            url_included: Whether to include source URLs in the response.
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.max_results = max_results
        self.url_included = url_included

        # Logger setup
        self.logger = logging.getLogger("WebSearch")
        self.logger.setLevel(logging.INFO)

        # Initialize OpenAI API client
        self.client = openai.OpenAI(api_key=self.api_key)

    def search(self, query: str) -> Tuple[str, Usage]:
        """
        Perform a web search and return the response.

        Args:
            query: User query string.

        Returns:
            Tuple containing:
            - response text (str)
            - token usage (Usage)
        """
        try:
            if self.url_included == True:
                query = query + " add" + str(self.max_results) + "url(s)"
            params = {
                "model": self.model_name,
                "tools": [{"type": "web_search_preview"}],
                "input": query,
            }

            response = self.client.responses.create(**params)

            # Extract the AI's response
            output_text = ""
            urls = []
            print(response.output)
            for output in response.output:
                if output.type == "message":
                    for content in output.content:
                        if content.type == "output_text":
                            output_text = content.text
                            annotations = content.annotations
                            if self.url_included:
                                number = 0
                                for single_url in annotations:
                                    if single_url.type == "url_citation":
                                        urls.append(f"{single_url.title}: {single_url.url}")
                                        number = number + 1
                                        if number == self.max_results:
                                            break

            if self.url_included and len(urls)!=0 :
                output_text += "\n\nSources:\n" + "\n".join(urls[:self.max_results])
                print(urls)

            # Extract token usage
            usage = Usage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                cached_prompt_tokens=response.usage.input_tokens_details.cached_tokens,
            )

            return output_text, usage

        except Exception as e:
            self.logger.error(f"Error during OpenAI Web Search API call: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    websearch_client = WebSearchClient(model_name="gpt-4o", url_included=True, max_results=2)

    query = "Explain to me how anthropic’s MCP works?"
    response_text, usage = websearch_client.search(query)

    print("Question:", query)
    print("Response:", response_text)
    print("Usage:", usage)

    websearch_client = WebSearchClient(model_name="gpt-4o-mini", url_included=False, max_results=0)
    query = "What are the principles of GRPO training?"
    response_text, usage = websearch_client.search(query)

    print("Question:", query)
    print("Response:", response_text)
    print("Usage:", usage)

