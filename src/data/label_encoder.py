from sklearn.preprocessing import LabelEncoder

class LabelEncoderFactory:
    def get():
        classes = ["nothing", "chrono", "grimpe", "lecture", "brossage"]

        encoder = LabelEncoder()

        encoder = encoder.fit(classes)

        return encoder
