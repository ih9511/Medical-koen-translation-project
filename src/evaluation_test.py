from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd

base_model_id = 'MLP-KTLim/llama-3-Korean-Bllossom-8B'
finetune_model_id = 'ih9511/llama-3-Korean-8B_koen_medical_translation'

finetune_model = AutoModelForCausalLM.from_pretrained(finetune_model_id, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

translation_pipeline = pipeline(
    'translation',
    model=finetune_model,
    tokenizer=tokenizer
)

translated_text = []
test_df = pd.read_csv('./data/processed_data/test_processed.csv')

for index, row in test_df[:1].iterrows():
    query = row['input']
    prompt = (
        "[[MEDICAL TRANSLATION MODE]] Translate this clinical text to Korean using:"
        "- Exact measurement units (preserve original numbers)"
        "- No idiomatic adaptations"
        f"English: {query}"
        "Korean:"
    )
    
    generated = translation_pipeline(
        prompt,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id
    )
    
    print(generated)
    generated_text = generated[0]['translation_text']
    
    output_start = generated_text.find("Korean:")
    
    if output_start != -1:
        translated_text.append(generated_text[output_start + len('Korean:'):].strip())
    else:
        translated_text.append(generated_text.strip())
        
    print(translated_text)