from models.ReasonEngine import ReasonEngine
import torch


if __name__ == "__main__":
    reason_engine = ReasonEngine(4, 8, 8, 100, 10, 50)
    # batch_size = 2
    query = [[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]],
             [[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5],[6,6,6,6,6,6,6,6]]]  # query: 2 * 3 * 8
    facts = [[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]],[[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5],[6,6,6,6,6,6,6,6]]],[[[1,1,1,1,1,1,1,1],[2,2,2,2,2,2,2,2],[3,3,3,3,3,3,3,3]],[[4,4,4,4,4,4,4,4],[5,5,5,5,5,5,5,5],[6,6,6,6,6,6,6,6]]]]  # 2 * 2 * 3 * 8
    query = torch.Tensor(query)
    facts = torch.Tensor(facts)
    scores = reason_engine(query, facts)
