import pandas as pd


class ImbalanceConstructor:

    def __init__(self, target_ratio: float, seed: int):
        self.target_ratio = target_ratio
        self.seed = seed

    def construct(
        self, df: pd.DataFrame, majority_class: str, minority_class: str
    ) -> pd.DataFrame:
        majority_df = df[df["class_name"] == majority_class]
        minority_df = df[df["class_name"] == minority_class]

        target_n = round(len(majority_df) / self.target_ratio)
        if target_n >= len(minority_df):
            raise ValueError(
                f"target_ratio too low for {len(minority_df)} images"
            )

        sampled_minority = minority_df.sample(n=target_n, random_state=self.seed)
        constructed = pd.concat([majority_df, sampled_minority], ignore_index=True)

        return constructed
