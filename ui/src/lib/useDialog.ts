import { useEffect, useRef } from "react";

/** Open dialogs, innermost last; only the top one reacts to Escape. */
const dialogStack: symbol[] = [];

/**
 * Modal behaviour for a dialog element: Escape closes it (topmost dialog only),
 * focus moves into it on open (`[data-autofocus]` or the dialog itself), and focus
 * returns to the previously focused element on close.
 *
 * Attach the returned ref to the element with `role="dialog"`. Pass `canClose: false`
 * while closing would be unsafe (for example during a save).
 */
export function useDialog<T extends HTMLElement = HTMLDivElement>(
  open: boolean,
  onClose: () => void,
  canClose = true,
) {
  const ref = useRef<T | null>(null);
  const onCloseRef = useRef(onClose);
  const canCloseRef = useRef(canClose);
  onCloseRef.current = onClose;
  canCloseRef.current = canClose;

  useEffect(() => {
    if (!open) {
      return;
    }
    const id = Symbol("dialog");
    dialogStack.push(id);
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;

    const node = ref.current;
    if (node && !node.contains(document.activeElement)) {
      const initial = node.querySelector<HTMLElement>("[data-autofocus]");
      if (initial) {
        initial.focus();
      } else {
        if (!node.hasAttribute("tabindex")) {
          node.setAttribute("tabindex", "-1");
        }
        node.focus({ preventScroll: true });
      }
    }

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || event.defaultPrevented) {
        return;
      }
      if (dialogStack[dialogStack.length - 1] !== id || !canCloseRef.current) {
        return;
      }
      event.preventDefault();
      onCloseRef.current();
    };
    document.addEventListener("keydown", onKeyDown);

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      const index = dialogStack.indexOf(id);
      if (index !== -1) {
        dialogStack.splice(index, 1);
      }
      if (previous && previous.isConnected) {
        previous.focus({ preventScroll: true });
      }
    };
  }, [open]);

  return ref;
}
