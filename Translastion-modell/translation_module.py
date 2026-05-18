from transformers import MarianMTModel, MarianTokenizer

def translate_de_to_en(text: str) -> dict:
    """
    Traduit un texte technique allemand en anglais en local avec Marian NMT.
    Retourne un dictionnaire prêt pour l'intégration dans le DMS.
    """
    if not text.strip():
        return {"original": "", "translated": "", "language": "en"}
        
    # Chargement du modèle spécifique Allemand -> Anglais
    model_name = "Helsinki-NLP/opus-mt-de-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    # Tokenisation et génération de la traduction
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    translated_tokens = model.generate(**inputs)
    
    # Décodage du résultat
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return {
        "original": text,
        "translated": translated_text,
        "language": "en"
    }

# --- TEST LOCAL ---
if __name__ == "__main__":
    sample_text = "Die Montage der Pumpenanlage muss gemäß der technischen Spezifikation erfolgen."
    result = translate_de_to_en(sample_text)
    print(f"Original : {result['original']}")
    print(f"Traduction : {result['translated']}")