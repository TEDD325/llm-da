import random
import os
import datetime
import time
import re
from faker import Faker
import json
from tqdm import tqdm


# 다국어 지원 설정
fake = Faker(['ko_KR', 'en_US', 'ja_JP'])
Faker.seed(42)  # 재현성을 위한 시드 설정

# 피드백 카테고리 및 비율 설정
CATEGORIES = {
    "기술적 문제": 0.30,  # 30%
    "불만 사항": 0.50,    # 50%
    "개선 요구": 0.10,    # 10%
    "장점": 0.10          # 10%
}

# 각 카테고리별 피드백 템플릿
FEEDBACK_TEMPLATES = {
    "기술적 문제": [
        "앱이 자주 충돌하는 문제가 있습니다. {specific_issue}. 이런 문제가 해결되면 좋겠습니다.",
        "음성 인식 기능이 {specific_issue} 상황에서 제대로 작동하지 않습니다. 기술적 개선이 필요합니다.",
        "번역 기능이 {specific_issue} 때 정확하지 않습니다. 특히 {language} 번역에서 문제가 자주 발생합니다.",
        "최근 업데이트 이후 {specific_issue} 문제가 생겼습니다. 이전 버전에서는 잘 작동했는데 아쉽습니다.",
        "앱 실행 시 로딩 시간이 {specific_issue}. 이 부분이 개선되었으면 합니다.",
        "배터리 소모가 {specific_issue}. 백그라운드에서도 많은 전력을 사용하는 것 같습니다.",
        "오프라인 모드에서 {specific_issue} 기능이 작동하지 않습니다. 인터넷 연결 없이도 기본 기능은 사용할 수 있으면 좋겠습니다.",
        "동영상 학습 자료가 {specific_issue} 문제로 제대로 재생되지 않습니다.",
        "알림 기능이 {specific_issue} 때문에 신뢰할 수 없습니다. 학습 일정을 관리하는데 어려움이 있습니다.",
        "데이터 동기화에 {specific_issue} 문제가 있습니다. 여러 기기에서 사용할 때 학습 진도가 제대로 반영되지 않습니다."
    ],
    "불만 사항": [
        "구독 가격이 너무 비싸다고 생각합니다. {specific_complaint}",
        "개인정보 처리 방식이 투명하지 않습니다. {specific_complaint}",
        "고객 지원 서비스가 {specific_complaint}. 문의에 대한 응답이 너무 늦습니다.",
        "무료 버전의 기능이 너무 제한적입니다. {specific_complaint}",
        "광고가 너무 많고 {specific_complaint}. 학습에 집중하기 어렵습니다.",
        "학습 콘텐츠의 질이 {specific_complaint}. 더 다양하고 깊이 있는 자료가 필요합니다.",
        "인터페이스가 {specific_complaint}. 메뉴 구조가 복잡하고 원하는 기능을 찾기 어렵습니다.",
        "학습 진도 추적 시스템이 {specific_complaint}. 내 학습 상태를 정확히 파악하기 어렵습니다.",
        "앱 디자인이 {specific_complaint}. 시각적으로 더 매력적이고 현대적인 디자인이 필요합니다.",
        "게임화 요소가 {specific_complaint}. 동기부여가 부족하고 지루함을 느낍니다."
    ],
    "개선 요구": [
        "더 다양한 언어 지원이 필요합니다. 특히 {language} 학습 자료가 추가되면 좋겠습니다.",
        "학습 진도에 따른 맞춤형 피드백이 {improvement_suggestion}.",
        "발음 교정 기능이 {improvement_suggestion}.",
        "단어장 관리 기능이 {improvement_suggestion}.",
        "다른 학습자들과의 소통 기능이 {improvement_suggestion}.",
        "실생활 대화 시나리오가 {improvement_suggestion}.",
        "문화적 배경 정보 제공이 {improvement_suggestion}.",
        "학습 목표 설정 및 관리 기능이 {improvement_suggestion}.",
        "오디오북 기능이 {improvement_suggestion}.",
        "AI 튜터의 개인화 수준이 {improvement_suggestion}."
    ],
    "장점": [
        "문법 교정 기능이 정말 유용합니다. {specific_praise}",
        "개인화된 학습 경로가 효과적입니다. {specific_praise}",
        "발음 코치 기능이 큰 도움이 됩니다. {specific_praise}",
        "게임화된 학습 방식이 동기부여에 좋습니다. {specific_praise}",
        "오프라인 학습 기능이 매우 편리합니다. {specific_praise}",
        "다양한 학습 자료가 제공되어 좋습니다. {specific_praise}",
        "직관적인 인터페이스가 사용하기 쉽습니다. {specific_praise}",
        "실시간 피드백 시스템이 학습에 효과적입니다. {specific_praise}",
        "문화적 맥락 설명이 언어 이해에 도움됩니다. {specific_praise}",
        "AI 튜터의 응답이 자연스럽고 도움이 됩니다. {specific_praise}"
    ]
}

# 세부 내용을 위한 데이터
SPECIFIC_ISSUES = [
    "메모리 누수로 인해", "서버 연결 오류로", "화면 렌더링 문제로", 
    "오디오 처리 지연으로", "데이터베이스 동기화 실패로", "API 호출 오류로",
    "캐시 관리 문제로", "네트워크 지연으로", "리소스 부족으로", "멀티스레딩 충돌로",
    "너무 길어서", "갑자기 느려져서", "자주 멈춰서", "배터리를 많이 소모해서",
    "메모리를 과도하게 사용해서", "데이터를 너무 많이 사용해서", "발열이 심해서",
    "시끄러운 환경", "백그라운드 실행 시", "여러 앱을 동시에 실행할 때",
    "특정 기기에서", "최신 OS 업데이트 후", "저사양 기기에서", "태블릿 모드에서"
]

SPECIFIC_COMPLAINTS = [
    "경쟁 앱에 비해 가성비가 떨어집니다", "학생 할인이 부족합니다", "장기 구독 혜택이 미미합니다",
    "개인 데이터가 어떻게 사용되는지 명확하지 않습니다", "필요 이상의 권한을 요구합니다",
    "데이터 수집 범위가 과도합니다", "프라이버시 정책이 복잡하고 이해하기 어렵습니다",
    "형식적이고 도움이 되지 않습니다", "응답 시간이 너무 깁니다", "자동응답만 제공합니다",
    "실제 도움을 받기 어렵습니다", "외국어 지원이 부족합니다", "문제 해결 능력이 떨어집니다",
    "기본 기능조차 제한적입니다", "무료 사용자를 홀대하는 느낌입니다", 
    "핵심 기능이 모두 유료입니다", "무료 체험 기간이 너무 짧습니다",
    "너무 자주 등장합니다", "관련성이 낮습니다", "학습을 방해합니다",
    "건너뛰기 어렵습니다", "유료 전환을 강요하는 느낌입니다", "지나치게 공격적입니다",
    "깊이가 부족합니다", "업데이트가 느립니다", "다양성이 부족합니다",
    "실용적이지 않습니다", "현지 문화를 반영하지 못합니다", "난이도 조절이 부적절합니다"
]

IMPROVEMENT_SUGGESTIONS = [
    "더 세밀하게 개선되면 좋겠습니다", "더 직관적으로 개선되면 좋겠습니다",
    "더 다양한 옵션이 추가되면 좋겠습니다", "사용자 정의 기능이 강화되면 좋겠습니다",
    "실시간으로 제공되면 좋겠습니다", "더 정확하게 개선되면 좋겠습니다",
    "AI 기술을 활용해 향상되면 좋겠습니다", "다른 앱과 연동되면 좋겠습니다",
    "클라우드 동기화가 지원되면 좋겠습니다", "오프라인에서도 작동하면 좋겠습니다",
    "더 쉽게 접근할 수 있으면 좋겠습니다", "더 시각적으로 표현되면 좋겠습니다",
    "게임화 요소가 추가되면 좋겠습니다", "소셜 기능이 통합되면 좋겠습니다",
    "더 개인화된 경험을 제공하면 좋겠습니다", "더 체계적으로 구성되면 좋겠습니다",
    "더 많은 예시가 포함되면 좋겠습니다", "실생활 적용이 강화되면 좋겠습니다",
    "더 다양한 난이도로 제공되면 좋겠습니다", "더 정기적으로 업데이트되면 좋겠습니다"
]

SPECIFIC_PRAISES = [
    "실시간으로 오류를 잡아주어 학습 효율이 크게 향상되었습니다",
    "제 실력과 관심사에 맞는 콘텐츠를 제공해 시간을 효율적으로 사용할 수 있습니다",
    "네이티브와 비교해 내 발음의 차이점을 명확히 보여주어 큰 도움이 됩니다",
    "포인트와 레벨 시스템이 학습 의욕을 지속적으로 유지하게 해줍니다",
    "여행 중에도 인터넷 없이 학습할 수 있어 매우 유용합니다",
    "비디오, 오디오, 텍스트 등 다양한 형식으로 학습할 수 있어 지루하지 않습니다",
    "복잡한 설명 없이도 쉽게 기능을 찾고 사용할 수 있습니다",
    "즉각적인 교정과 조언이 학습 곡선을 가속화합니다",
    "단순 번역을 넘어 표현의 배경과 용례를 이해하는 데 도움됩니다",
    "마치 실제 교사와 대화하는 것 같은 자연스러운 상호작용이 가능합니다",
    "꾸준히 사용하면서 실력이 눈에 띄게 향상되고 있습니다",
    "다른 앱들과 달리 실용적인 표현 위주로 가르쳐 실생활에 바로 적용할 수 있습니다",
    "친구들에게도 추천했고 모두 만족하고 있습니다",
    "직장에서 외국어 사용 능력이 향상되어 업무 효율이 높아졌습니다",
    "여행 중 현지인과 소통하는 데 큰 도움이 되었습니다"
]

LANGUAGES = ["영어", "일본어", "중국어", "스페인어", "프랑스어", "독일어", "러시아어", "이탈리아어", "포르투갈어", "아랍어", "힌디어", "한국어"]

# 다양한 어투를 위한 문장 종결 표현
SENTENCE_ENDINGS = [
    "요.", "용", "습니다.", "어요.", "어요....", "네요.", "군요.", "네.", "죠.", 
    "다고 생각합니다.", "는 것 같아요.", "는 느낌이 듭니다.", 
    "는 점이 아쉽습니다.", "면 좋겠습니다.", "기를 바랍니다.",
    "길 희망합니다.", "다고 봅니다.", "는 것이 사실입니다."
]

# 다양한 어투를 위한 접두사
PREFIXES = [
    "제 생각에는 ", "솔직히 말해서 ", "개인적으로는 ", "사용해본 결과 ", 
    "경험상 ", "정말 ", "확실히 ", "분명히 ", "아무래도 ", "아쉽게도 ",
    "다행히 ", "놀랍게도 ", "기대와 달리 ", "예상대로 ", "결론적으로 ",
    "",   # 빈 문자열도 포함하여 접두사가 없는 경우도 생성
]

def generate_random_date():
    """2024년 1월 1일부터 2025년 3월 4일 사이의 랜덤 날짜 생성"""
    start_date = datetime.date(2024, 1, 1)
    end_date = datetime.date(2025, 3, 4)
    days_between = (end_date - start_date).days
    random_days = random.randint(0, days_between)
    return start_date + datetime.timedelta(days=random_days)

def apply_style_variation(text):
    """텍스트에 다양한 어투 적용"""
    sentences = text.split('. ')
    styled_sentences = []
    
    for sentence in sentences:
        if not sentence:  # 빈 문자열 처리
            continue
            
        # 마침표가 이미 있는지 확인
        has_period = sentence[-1] in ['.', '?', '!']
        clean_sentence = sentence[:-1] if has_period else sentence
        
        # 50% 확률로 접두사 추가
        if random.random() < 0.5:
            prefix = random.choice(PREFIXES)
            clean_sentence = prefix + clean_sentence[0].lower() + clean_sentence[1:]
        
        # 문장 종결 표현 변경 (한국어 문장인 경우에만)
        if any('\u3131' <= c <= '\u318F' or '\uAC00' <= c <= '\uD7A3' for c in clean_sentence):
            # 기존 종결어미 제거 시도
            for ending in ["습니다", "합니다", "입니다", "어요", "아요", "해요", "네요", "군요"]:
                if clean_sentence.endswith(ending):
                    clean_sentence = clean_sentence[:-len(ending)]
                    break
            
            # 새 종결어미 추가
            ending = random.choice(SENTENCE_ENDINGS)
            if clean_sentence.endswith("다"):
                clean_sentence = clean_sentence[:-1]
            styled_sentences.append(clean_sentence + ending)
        else:
            # 한국어가 아닌 경우 원래 문장 사용
            styled_sentences.append(sentence)
    
    return ' '.join(styled_sentences)

def generate_feedback(category):
    """특정 카테고리의 피드백 생성"""
    template = random.choice(FEEDBACK_TEMPLATES[category])
    
    if category == "기술적 문제":
        specific_issue = random.choice(SPECIFIC_ISSUES)
        language = random.choice(LANGUAGES)
        feedback = template.format(specific_issue=specific_issue, language=language)
    
    elif category == "불만 사항":
        specific_complaint = random.choice(SPECIFIC_COMPLAINTS)
        feedback = template.format(specific_complaint=specific_complaint)
    
    elif category == "개선 요구":
        improvement_suggestion = random.choice(IMPROVEMENT_SUGGESTIONS)
        language = random.choice(LANGUAGES)
        feedback = template.format(improvement_suggestion=improvement_suggestion, language=language)
    
    elif category == "장점":
        specific_praise = random.choice(SPECIFIC_PRAISES)
        feedback = template.format(specific_praise=specific_praise)
    
    # 어투 변형 적용
    return apply_style_variation(feedback)

def generate_rating(category):
    """카테고리에 따른 적절한 평점 생성"""
    if category == "장점":
        return random.randint(4, 5)
    elif category == "개선 요구":
        return random.randint(3, 4)
    elif category == "기술적 문제":
        return random.randint(2, 3)
    elif category == "불만 사항":
        return random.randint(1, 2)
    return random.randint(1, 5)

def create_feedback_file(feedback_data, file_path):
    """피드백 데이터를 파일로 저장"""
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(f"제품명: AI 언어 학습 앱\n")
        f.write(f"평점: {feedback_data['rating']}/5\n")
        f.write(f"날짜: {feedback_data['date']}\n")
        f.write(f"피드백: \n")
        f.write(f"{feedback_data['content']}\n")

def generate_bulk_feedback(count=10000, output_dir=None, batch_size=500):
    # Set default output directory to be in the same directory as this script
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "review_data")
    """대량의 피드백 데이터를 생성하고 바로 파일로 저장"""
    # 출력 디렉토리 생성 (이미 존재하면 그대로 사용)
    os.makedirs(output_dir, exist_ok=True)
    
    # 기존 파일 확인 및 다음 파일 번호 결정
    existing_files = [f for f in os.listdir(output_dir) 
                     if f.startswith("customer_feedback_") and f.endswith(".txt")]
    
    # 기존 파일 번호 추출
    existing_numbers = []
    for filename in existing_files:
        try:
            num = int(filename.replace("customer_feedback_", "").replace(".txt", ""))
            existing_numbers.append(num)
        except ValueError:
            continue
    
    # 시작 번호 결정 (기존 파일이 없으면 1부터 시작)
    start_number = 1
    if existing_numbers:
        start_number = max(existing_numbers) + 1
        
    print(f"기존 파일 {len(existing_numbers)}개 발견, 파일 번호 {start_number}부터 시작합니다.")
    
    # 카테고리별 개수 계산
    category_counts = {}
    remaining = count
    
    for category, ratio in list(CATEGORIES.items())[:-1]:  # 마지막 카테고리는 남은 수로 계산
        category_count = int(count * ratio)
        category_counts[category] = category_count
        remaining -= category_count
    
    # 마지막 카테고리에 남은 수 할당
    last_category = list(CATEGORIES.keys())[-1]
    category_counts[last_category] = remaining
    
    # 생성할 피드백 순서 결정 (카테고리 기반)
    feedback_categories = []
    for category, category_count in category_counts.items():
        feedback_categories.extend([category] * category_count)
    
    # 순서 섞기
    random.shuffle(feedback_categories)
    
    # 진행 상황 표시를 위한 tqdm 설정
    print(f"\n{count}개의 피드백 데이터 생성 및 저장 중...")
    
    # 배치 단위로 처리
    total_batches = (count + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="피드백 생성 진행률"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, count)
        batch_categories = feedback_categories[start_idx:end_idx]
        
        # 현재 배치의 피드백 생성 및 저장
        for i, category in enumerate(batch_categories):
            date = generate_random_date().strftime("%Y-%m-%d")
            content = generate_feedback(category)
            rating = generate_rating(category)
            
            feedback_data = {
                "category": category,
                "date": date,
                "content": content,
                "rating": rating
            }
            
            # 개별 파일로 저장 (시작 번호부터 순차적으로 증가)
            file_number = start_number + start_idx + i
            file_path = os.path.join(output_dir, f"customer_feedback_{file_number}.txt")
            create_feedback_file(feedback_data, file_path)
    
    return count

def save_category_stats(output_dir=None):
    # Use the same directory resolution logic as generate_bulk_feedback
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "review_data")
    """생성된 피드백의 카테고리별 통계 저장"""
    # 디렉토리 내 모든 피드백 파일 검사
    category_counts = {category: 0 for category in CATEGORIES.keys()}
    total_files = 0
    
    # 피드백 파일만 필터링
    feedback_files = [f for f in os.listdir(output_dir) 
                     if f.startswith("customer_feedback_") and f.endswith(".txt")]
    
    # 샘플링하여 카테고리 분포 확인 (모든 파일을 검사하면 너무 오래 걸림)
    sample_size = min(1000, len(feedback_files))
    if feedback_files:
        sample_files = random.sample(feedback_files, sample_size)
    else:
        sample_files = []
    
    for filename in sample_files:
        file_path = os.path.join(output_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 파일에서 카테고리 정보 추출
                category_found = False
                
                # 1. 카테고리 키워드 직접 찾기
                for category in CATEGORIES.keys():
                    if category in content:
                        category_counts[category] += 1
                        category_found = True
                        break
                
                # 2. 평점을 기반으로 카테고리 추정
                if not category_found:
                    # 평점 추출 시도
                    rating_match = re.search(r'평점:\s*(\d+)/5', content)
                    if rating_match:
                        rating = int(rating_match.group(1))
                        if rating <= 2:
                            category_counts["불만 사항"] += 1
                        elif rating == 3:
                            category_counts["개선 요구"] += 1
                        else:  # rating >= 4
                            category_counts["장점"] += 1
                    else:
                        # 기본값
                        category_counts["기술적 문제"] += 1
                
                total_files += 1
        except Exception as e:
            print(f"파일 {filename} 처리 중 오류 발생: {e}")
    
    # 전체 피드백 파일 수 계산
    all_files_count = len(feedback_files)
    
    # 결과 저장
    stats = {
        "total_files": all_files_count,
        "sampled_files": total_files,
        "category_distribution": {}
    }
    
    # 비율 계산
    for category, count in category_counts.items():
        if total_files > 0:
            percentage = round(count / total_files * 100, 2)
        else:
            percentage = 0.0
            
        stats["category_distribution"][category] = {
            "count": count,
            "percentage": percentage
        }
    
    # 통계 파일 저장 (덮어쓰기)
    stats_file = os.path.join(output_dir, "category_stats.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"카테고리 통계가 {stats_file}에 저장되었습니다.")
    return stats

# 실행 코드
if __name__ == "__main__":
    # 시작 시간 기록
    start_time = time.time()
    
    # 스크립트 디렉토리를 기준으로 출력 디렉토리 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "review_data")
    
    # 기존 디렉토리 확인 (경고 없이 그대로 사용)
    if not os.path.exists(output_dir):
        print(f"{output_dir} 디렉토리를 생성합니다.")
    elif os.listdir(output_dir):
        print(f"{output_dir} 디렉토리가 이미 존재하며, 기존 파일을 유지합니다.")
        print("새로운 파일은 기존 파일 번호 이후부터 생성됩니다.")
    
    # 10,000개의 피드백 데이터를 바로 파일로 생성
    total_count = generate_bulk_feedback(count=100, output_dir=output_dir)
    
    # 카테고리 통계 저장
    stats = save_category_stats(output_dir)
    
    # 소요 시간 계산
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    print(f"\n작업 완료! {total_count}개의 피드백 데이터가 {output_dir} 디렉토리에 생성되었습니다.")
    print(f"소요 시간: {int(minutes)}분 {seconds:.2f}초")
    
    # 카테고리 분포 출력
    print("\n카테고리 분포 (샘플 기준):")
    for category, data in stats["category_distribution"].items():
        print(f"  - {category}: {data['percentage']}% ({data['count']}개)")