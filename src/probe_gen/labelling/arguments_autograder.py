import json
import re


def _extract_answer(text):
    """
    Extract a number (0-10) from patterns like "X out of 10" or "X/10" in the input string.
    
    Args:
        text (str): Input string to search
        
    Returns:
        int or str: The extracted number if found, otherwise the original input string
    """

    explicit_patterns = [
        r"I\s+would\s+rate\s+this\s+argument\s+(?:a\s+)?\*\*?([0-9]|10)\*\*?"
        #r"I\s+would\s+rate\s+this\s+argument\s+(?:a\s+)?(?:\*\*([0-9]|10)\*\*|([0-9]|10))",
        r"Rating:\s*([0-9]|10)"
    ]

    for pattern in explicit_patterns:
        explicit_match = re.search(pattern, text, re.IGNORECASE)
        if explicit_match:
            return explicit_match.group(1)

    # Pattern to match both "number out of 10" and "number/10" where number is 0-10
    pattern = r'\b([0-9]|10)(?:\s+out\s+of\s+10|/10)\b'
    
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1)
    else:
        return text

def label_and_save_dataset_arguments(
    responses_file, out_file, num_balanced=4000, include_non_determined=False
):
    # Load outputs
    outputs = []
    output_lines = []
    with open(responses_file, "r") as file:
        for line in file:
            data = json.loads(line)
            inputs = json.loads(data["inputs"])

            assistant = inputs[1]["content"]
            extracted_answer = _extract_answer(assistant)
            outputs.append(extracted_answer)
            output_lines.append(line)

    def is_valid_answer(o):
        valid_responses = ['0','1','2','3','4','5','6','7','8','9','10']
        return o in valid_responses

    num_negative = 0
    num_positive = 0
    with open(out_file, "w") as file:
        for i in range(0, len(outputs), 3):
            if is_valid_answer(outputs[i]) and is_valid_answer(outputs[i + 1]) and is_valid_answer(outputs[i + 2]):
                control = int(outputs[i])
                pos = int(outputs[i+1])
                neg = int(outputs[i+2])
                if control == pos:
                    # Negative
                    if num_negative < num_balanced / 2:
                        file.write(
                            f'{output_lines[i+1][:-2]},"scale_labels":10'
                            + "}\n"
                        )
                        num_negative += 1
                elif control + 1 <= pos:
                    # Very positive
                    if num_positive < num_balanced / 2:
                        file.write(
                            f'{output_lines[i+1][:-2]},"scale_labels":1'
                            + "}\n"
                        )
                        num_positive += 1
                elif include_non_determined:
                    file.write(
                        f'{output_lines[i+1][:-2]},"scale_labels":null'
                        + "}\n"
                    )
                if control == neg:
                    # Negative
                    if num_negative < num_balanced / 2:
                        file.write(
                            f'{output_lines[i+2][:-2]},"scale_labels":10'
                            + "}\n"
                        )
                        num_negative += 1
                elif control - 1 >= neg:
                    # Very positive
                    if num_positive < num_balanced / 2:
                        file.write(
                            f'{output_lines[i+2][:-2]},"scale_labels":1'
                            + "}\n"
                        )
                        num_positive += 1
                elif include_non_determined:
                    file.write(
                        f'{output_lines[i+2][:-2]},"scale_labels":null'
                        + "}\n"
                    )
            elif include_non_determined:
                file.write(
                    f'{output_lines[i+1][:-2]},"scale_labels":null'
                    + "}\n"
                )
                file.write(
                    f'{output_lines[i+2][:-2]},"scale_labels":null'
                    + "}\n"
                )

    print(num_negative)
    print(num_positive)