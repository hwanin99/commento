import pandas as pd
from openai import OpenAI
from PIL import Image
from IPython.display import display


client = OpenAI(api_key='Change to your OpenAI API Key')

# GPT로 텍스트 수정
def getTextFromGPT(prompt):
    completion = client.chat.completions.create(
        model="gpt-4o",
        max_tokens = 256,
        temperature = 0.7,
        top_p = 1.0,
        frequency_penalty = 0.1,
        presence_penalty = 0.0,
        messages=[
            {"role": "system",
            "content": "Modify the given caption by prepending 'Turn it into' and refining the wording to make it more natural and descriptive, while ensuring it remains a single concise sentence. Only return the modified caption without any additional formatting or explanation."
  },
            {
                "role": "user",
                "content":f"{prompt}",
            },
        ],
    )

    response = completion.choices[0].message.content
    return response


# 시각화 함수
def visualize(image_path):
    image = Image.open(image_path).convert('RGB')
    thumb = image.copy()
    thumb.thumbnail([256, 256])
    display(thumb)


# 전체 파이프라인 함수
def save_modified_captions(csv_path, img_root, output_file):
    df = pd.read_csv(csv_path)

    modified_prompts = []

    for idx, row in df.iterrows():
        image_path = f"{img_root}/{row['image']}"
        original_prompt = row['prompt']

        print(f"\n[{idx+1}/{len(df)}] Processing: {row['image']}")
        print("Original:", original_prompt)

        # GPT로 캡션 수정
        modified_caption = getTextFromGPT(original_prompt)
        modified_prompts.append(modified_caption)

        # 이미지 시각화
        visualize(image_path)

        # 결과 출력
        print("Modified:", modified_caption)

    # 수정된 caption 저장
    df['modified_prompt'] = modified_prompts
    df.to_csv(f'./dataset/{output_file}', index=False)
    print(f"\n Modified captions saved to '{output_file}'")
