export default function apply(patch: EditPatch): EditReport {
return host.ide.edit(patch);
}
