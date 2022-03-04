import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
# sys.path += ['../']
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../')))
from gerbil_experiments.nn_processing import Annotator


class GetHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers["Content-Length"])
        post_data = self.rfile.read(content_length)
        self.send_response(200)
        self.end_headers()
        doc_text = read_json(post_data)
        # try:
        response = annotator.get_predicts(doc_text)

        print("*" * 20)
        print("this is the sentence")
        print(doc_text)

        print("response in server.py code:\n", response)
        self.wfile.write(bytes(json.dumps(response), "utf-8"))
        return


def read_json(post_data):
    data = json.loads(post_data.decode("utf-8"))
    # print("received data:", data)
    text = data["text"]
    # spans = [(int(j["start"]), int(j["length"])) for j in data["spans"]]
    return text


class Tee:
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()


def terminate():
    tee.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str,
                        help='log path')
    parser.add_argument('--blink_dir', type=str,
                        help='blink pretrained bi-encoder path')
    parser.add_argument(
        "--passage_len", type=int, default=32,
        help="the length of each passage"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="length of stride when chunking passages",
    )
    parser.add_argument('--bsz_retriever', type=int, default=4096,
                        help='the batch size of retriever')
    parser.add_argument('--max_len_retriever', type=int, default=42,
                        help='max length of the retriever input passage ')
    parser.add_argument('--retriever_path', type=str,
                        help='trained retriever path')
    parser.add_argument('--type_retriever_loss', type=str,
                        default='sum_log_nce',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='type of marginalize for retriever')
    parser.add_argument('--gpus', default='', type=str,
                        help='GPUs separated by comma [%(default)s]')
    parser.add_argument('--cands_embeds_path', type=str,
                        help='the path of candidates embeddings')
    parser.add_argument('--k', type=int, default=100,
                        help='top-k candidates for retriever')
    parser.add_argument('--ents_path', type=str,
                        help='entity file path')
    parser.add_argument('--max_len_reader', type=int, default=180,
                        help='max length of joint input [%(default)d]')
    parser.add_argument('--max_num_candidates', type=int, default=100,
                        help='max number of candidates [%(default)d] when '
                             'eval for reader')
    parser.add_argument('--bsz_reader', type=int, default=32,
                        help='batch size [%(default)d]')
    parser.add_argument('--reader_path', type=str,
                        help='trained reader path')
    parser.add_argument('--type_encoder', type=str,
                        default='squad2_electra_large',
                        help='the type of encoder')
    parser.add_argument('--type_span_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--type_rank_loss', type=str,
                        default='sum_log',
                        choices=['log_sum', 'sum_log', 'sum_log_nce',
                                 'max_min'],
                        help='the type of marginalization for reader')
    parser.add_argument('--num_spans', type=int, default=3,
                        help='top num_spans for evaluation on reader')
    parser.add_argument('--thresd', type=float, default=0.05,
                        help='probabilty threshold for evaluation on reader')
    parser.add_argument('--max_answer_len', type=int, default=10,
                        help='max length of answer [%(default)d]')
    parser.add_argument('--max_passage_len', type=int, default=32,
                        help='max length of question [%(default)d] for reader')
    parser.add_argument('--document', type=str,
                        help='test document')
    parser.add_argument('--save_span_path', type=str,
                        help='save span-based document-level results path')
    parser.add_argument('--save_char_path', type=str,
                        help='save char-based path')
    parser.add_argument('--add_topic', action='store_true',
                        help='add title?')
    parser.add_argument('--do_rerank', action='store_true',
                        help='do reranking for reader?')
    parser.add_argument('--use_title', action='store_true',
                        help='use title?')
    parser.add_argument('--no_multi_ents', action='store_true',
                        help='no repeated entities are allowed given a span?')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    annotator = Annotator(args)
    server = HTTPServer(("localhost", 5555), GetHandler)
    print("Starting server at http://localhost:5555")

    tee = Tee("server.txt", "w")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        terminate()
        exit(0)
