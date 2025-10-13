using System;
using System.IO;

namespace LayoutSdk.Tests;

internal static class TestModelFiles
{
    private const string OnnxBase64 = "CAcSBHRlc3Q6WwoQCgFYEgFZIghJZGVudGl0eRINSWRlbnRpdHlHcmFwaFobCgFYEhYKFAgBEhAKAggBCgIIBAoCCAIKAggCYhsKAVkSFgoUCAESEAoCCAEKAggECgIIAgoCCAJCBAoAEAk=";
    private const string OpenVinoXmlBase64 = "PD94bWwgdmVyc2lvbj0iMS4wIj8+CjxuZXQgbmFtZT0iSWRlbnRpdHkiIHZlcnNpb249IjExIj4KCTxsYXllcnM+CgkJPGxheWVyIGlkPSIwIiBuYW1lPSJpbnB1dCIgdHlwZT0iUGFyYW1ldGVyIiB2ZXJzaW9uPSJvcHNldDEiPgoJCQk8ZGF0YSBzaGFwZT0iMSw0LDIsMiIgZWxlbWVudF90eXBlPSJmMzIiIC8+CgkJCTxvdXRwdXQ+CgkJCQk8cG9ydCBpZD0iMCIgcHJlY2lzaW9uPSJGUDMyIiBuYW1lcz0iaW5wdXQiPgoJCQkJCTxkaW0+MTwvZGltPgoJCQkJCTxkaW0+NDwvZGltPgoJCQkJCTxkaW0+MjwvZGltPgoJCQkJCTxkaW0+MjwvZGltPgoJCQkJPC9wb3J0PgoJCQk8L291dHB1dD4KCQk8L2xheWVyPgoJCTxsYXllciBpZD0iMSIgbmFtZT0icmVzdWx0IiB0eXBlPSJSZXN1bHQiIHZlcnNpb249Im9wc2V0MSI+CgkJCTxpbnB1dD4KCQkJCTxwb3J0IGlkPSIwIiBwcmVjaXNpb249IkZQMzIiPgoJCQkJCTxkaW0+MTwvZGltPgoJCQkJCTxkaW0+NDwvZGltPgoJCQkJCTxkaW0+MjwvZGltPgoJCQkJCTxkaW0+MjwvZGltPgoJCQkJPC9wb3J0PgoJCQk8L2lucHV0PgoJCTwvbGF5ZXI+Cgk8L2xheWVycz4KCTxlZGdlcz4KCQk8ZWRnZSBmcm9tLWxheWVyPSIwIiBmcm9tLXBvcnQ9IjAiIHRvLWxheWVyPSIxIiB0by1wb3J0PSIwIiAvPgoJPC9lZGdlcz4KCTxydF9pbmZvIC8+CjwvbmV0Pgo=";

    public static string CreateOnnxModelFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"model-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, Convert.FromBase64String(OnnxBase64));
        return path;
    }

}
