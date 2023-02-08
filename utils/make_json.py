import json


def make_json_file(args, cpt, score):
    result = {"version": "1.0.0.0", "concepts": [], "contents": []}
    for i in range(len(cpt)):
        result["concepts"].append({"id": i, "name": "name" + str(i), "activation": int(cpt[i] * 100)})
        result["contents"].append({"id": i, "name": "name" + str(i), "score": int(score[i] * 100)})

    final = {"inference_result": result}
    f2 = open(args.inference_result_dir + "/concept_results.json", "w")
    json.dump(final, f2)
    f2.close()