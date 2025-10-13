namespace LayoutSdk.Factories;

public interface ILayoutBackendFactory
{
    ILayoutBackend Create(LayoutRuntime runtime);
}
